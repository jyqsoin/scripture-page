# -*- coding: utf-8 -*-
"""独立单股环境与随机选股的联合多股环境。"""

import numpy as np


class TradingEnvV2:
    def __init__(self, view, config):
        self.view = view
        self.cfg = config
        self.lookback = int(config.get("lookback", 60))
        self.feature_count = int(config.get("feature_count", 20))
        self.commission = float(config.get("commission", 0.0006))
        self.slippage = float(config.get("slippage", 0.001))
        self.stamp_duty = float(config.get("stamp_duty", 0.0005))
        self.trade_threshold = float(config.get("trade_threshold", 0.03))
        self.max_position = float(config.get("max_position", 1.0))
        self.liquidate_on_done = bool(config.get("liquidate_on_done", True))
        if len(view) <= self.lookback + 1:
            raise ValueError("%s 数据不足以构造 60 天状态" % view.code)
        self.initial_cash = float(config.get("initial_cash", 1_000_000))
        # 只改变执行与风控参数，不触碰奖励函数；300750 可在 risk_overrides 中单独收紧。
        self.code = str(getattr(view, "code", ""))
        self.risk = self._risk_profile(config)
        self.max_position = float(self.risk["max_position"])
        self.reset()

    def _risk_profile(self, config):
        profile = {
            "max_position": float(config.get("max_position", 1.0)),
            "stop_loss": float(config.get("stop_loss", 0.15)),
            "max_hold_days": int(config.get("max_hold_days", 20)),
            "timeout_close_fraction": float(config.get("timeout_close_fraction", 1.0)),
            "residual_close_fraction": float(config.get("residual_close_fraction", 0.05)),
            "rebalance_days": int(config.get("rebalance_days", 1)),
            "cooldown_after_exit": int(config.get("cooldown_after_exit", 0)),
            "trailing_stop": float(config.get("trailing_stop", 0.0)),
            "trailing_arm": float(config.get("trailing_arm", 0.0)),
            "dd_breaker": float(config.get("dd_breaker", 1.0)),
            "dd_breaker_cooldown": int(config.get("dd_breaker_cooldown", 0)),
            "trend_entry_required": bool(config.get("trend_entry_required", False)),
            "ma_exit_mode": str(config.get("ma_exit_mode", "both")),
            "trend_participation_floor": float(config.get("trend_participation_floor", 0.0)),
            "pl_gate_min_trades": int(config.get("pl_gate_min_trades", 5)),
            "pl_gate_ratio": float(config.get("pl_gate_ratio", 0.0)),
            "pl_gate_scale": float(config.get("pl_gate_scale", 0.5)),
        }
        overrides = config.get("risk_overrides", {}) or {}
        for key, value in (overrides.get(self.code, {}) or {}).items():
            if key in profile:
                profile[key] = type(profile[key])(value)
        return profile

    def _state_at(self, index):
        sequence = self.view.features[index - self.lookback + 1:index + 1]
        return np.ascontiguousarray(sequence, dtype=np.float32)

    def _state(self):
        account = np.array([
            self.cash / max(self.nav * self.initial_cash, 1e-8),
            self.position_fraction,
            self.nav,
            self.drawdown,
            self.last_action,
            min(self.hold_steps / 60.0, 3.0),
            np.clip(self.unrealized_pnl, -1, 1),
            np.clip(self.return_history[-1] if self.return_history else 0.0, -0.2, 0.2) * 5,
        ], dtype=np.float32)
        state = np.concatenate([self._state_at(self.index).reshape(-1), account])
        self.last_state = state
        return state

    def reset(self):
        self.index = self.lookback - 1
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position_fraction = 0.0
        self.nav = 1.0
        self.peak_nav = 1.0
        self.drawdown = 0.0
        self.last_action = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0
        self.return_history = []
        self.nav_history = [1.0]
        self.position_history = [0.0]
        self.date_history = [self.view.dates[self.index]]
        self.price_history = [float(self.view.close[self.index])]
        self.trade_returns = []
        self.trade_log = []
        self.trade_count = 0
        self.losing_streak = 0
        self.entry_value = 0.0
        self.entry_date = None
        self.cycle_cost = 0.0
        self.cycle_proceeds = 0.0
        self.peak_unrealized = 0.0
        self.days_since_order = 10 ** 6
        self.entry_block_days = 0
        self.total_fee = 0.0
        self.done = False
        self.last_state = self._state()
        return self.last_state

    def _profit_loss_ratio(self):
        if not self.trade_returns:
            return 1.0
        values = np.asarray(self.trade_returns)
        wins, losses = values[values > 0], values[values < 0]
        if len(losses) == 0:
            return 5.0 if len(wins) else 1.0
        if len(wins) == 0:
            return 0.0
        return float(np.clip((wins.mean() + 1e-8) / (abs(losses.mean()) + 1e-8), 0.0, 5.0))

    def _is_high_volatility(self, index=None):
        index = self.index if index is None else index
        volatility = float(self.view.volatility60[index])
        median = float(self.view.volatility_median60[index])
        return bool(np.isfinite(volatility) and np.isfinite(median) and volatility > median)

    def _adaptive_drawdown_scale(self, high_volatility):
        if high_volatility:
            return float(self.cfg.get("high_vol_drawdown_multiplier", 1.5))
        return 1.0

    def _sell(self, quantity, price, reason):
        """所有卖出统一由此记账；只有持仓归零才形成一笔完整平仓。"""
        quantity = min(float(quantity), max(self.shares, 0.0))
        if quantity <= 1e-10 or self.shares <= 1e-10:
            return 0.0, False
        old_shares = self.shares
        old_basis = self.entry_value
        execution_price = price * (1 - self.slippage)
        proceeds = quantity * execution_price
        fee = proceeds * (self.commission + self.stamp_duty)
        net_proceeds = proceeds - fee
        self.cash += net_proceeds
        self.shares -= quantity
        self.cycle_proceeds += net_proceeds
        self.entry_value = old_basis * max(1 - quantity / old_shares, 0.0)
        self.trade_count += 1
        closed_now = False

        if self.shares <= 1e-8:
            self.shares = 0.0
            self.entry_value = 0.0
            if self.cycle_cost > 0:
                trade_return = (self.cycle_proceeds - self.cycle_cost) / self.cycle_cost
                self.trade_returns.append(float(trade_return))
                self.trade_log.append({
                    "entry_date": str(self.entry_date)[:10],
                    "exit_date": str(self.view.dates[self.index])[:10],
                    "net_return": float(trade_return),
                    "exit_reason": reason,
                })
                if trade_return < 0:
                    self.losing_streak += 1
                else:
                    self.losing_streak = 0
                closed_now = True
            self.cycle_cost = 0.0
            self.cycle_proceeds = 0.0
            self.entry_date = None
            self.peak_unrealized = 0.0
            self.entry_block_days = max(self.entry_block_days, int(self.risk["cooldown_after_exit"]))
        return fee, closed_now

    def step(self, raw_action):
        if self.done:
            raise RuntimeError("当前股票本 episode 已结束，请 reset 后再交互")
        cfg = self.cfg
        current_index = self.index
        is_final = current_index >= len(self.view) - 1
        action = float(np.clip(np.asarray(raw_action).reshape(-1)[0], -4, 4))
        # A 股现货环境仅做多；负动作表示空仓，不产生无法记账的虚拟空头。
        target = float(np.clip(np.tanh(action), 0.0, self.max_position))
        price = float(self.view.close[current_index])
        portfolio = self.cash + self.shares * price
        current_fraction = self.shares * price / max(portfolio, 1e-8)
        average_cost = self.entry_value / self.shares if self.shares > 0 and self.entry_value > 0 else 0.0
        current_unrealized = price / average_cost - 1 if average_cost > 0 else 0.0
        trend_active = price > self.view.ma20[current_index] and self.view.momentum20[current_index] > 0
        downtrend_active = price < self.view.ma20[current_index] and self.view.momentum20[current_index] < 0
        high_volatility = self._is_high_volatility(current_index)
        if self.entry_block_days > 0:
            self.entry_block_days -= 1
        if self.shares > 0:
            self.peak_unrealized = max(self.peak_unrealized, current_unrealized)
        else:
            self.peak_unrealized = 0.0
        risk = self.risk
        price_broke_ma = price < self.view.ma20[current_index]
        ma_exit = price_broke_ma if risk["ma_exit_mode"] == "price_below_ma" else downtrend_active

        # 强制规则优先级：期末/回撤熔断 > 硬止损 > 移动止盈 > 均线破位 > 超时全平 > 残余仓位强平。
        forced_reason = ""
        if is_final and self.liquidate_on_done and self.shares > 0:
            target, forced_reason = 0.0, "final_liquidation"
        elif self.shares > 0 and self.drawdown >= float(risk["dd_breaker"]):
            target, forced_reason = 0.0, "drawdown_breaker"
            self.entry_block_days = max(self.entry_block_days, int(risk["dd_breaker_cooldown"]))
        elif self.shares > 0 and current_unrealized < -float(risk["stop_loss"]):
            target, forced_reason = 0.0, "stop_loss"
        elif (
            self.shares > 0
            and float(risk["trailing_stop"]) > 0
            and self.peak_unrealized >= float(risk["trailing_arm"])
            and self.peak_unrealized - current_unrealized >= float(risk["trailing_stop"])
        ):
            target, forced_reason = 0.0, "trailing_stop"
        elif self.shares > 0 and ma_exit:
            target, forced_reason = 0.0, (
                "ma_break_exit" if risk["ma_exit_mode"] == "price_below_ma" else "downtrend_exit"
            )
        elif (
            self.shares > 0
            and self.hold_steps >= int(risk["max_hold_days"])
            and current_unrealized < float(cfg.get("min_profit_target", 0.05))
        ):
            fraction = float(risk["timeout_close_fraction"])
            target = 0.0 if fraction >= 1.0 else current_fraction * (1 - fraction)
            forced_reason = "timeout_close"
        elif self.shares > 0 and self.hold_steps >= 2 * int(risk["max_hold_days"]):
            target, forced_reason = 0.0, "max_hold_hard_close"
        elif (
            self.shares > 0
            and 0 < current_fraction <= float(risk["residual_close_fraction"])
            and target < current_fraction
        ):
            target, forced_reason = 0.0, "residual_close"

        if not forced_reason and 0.0 < target <= float(risk["residual_close_fraction"]):
            # 残余目标直接归零，避免 1%~5% 的尾仓永远无法形成完整平仓。
            target = 0.0
            if current_fraction > 0:
                forced_reason = "target_snap_close"
        if not forced_reason:
            if self.entry_block_days > 0 and target > current_fraction:
                target = current_fraction
            if risk["trend_entry_required"] and target > current_fraction and not trend_active:
                target = current_fraction
            floor = float(risk["trend_participation_floor"])
            if floor > 0 and trend_active and self.entry_block_days == 0 and target < floor:
                target = min(floor, self.max_position)
            if (
                len(self.trade_returns) >= int(risk["pl_gate_min_trades"])
                and float(risk["pl_gate_ratio"]) > 0
                and self._profit_loss_ratio() < float(risk["pl_gate_ratio"])
            ):
                target = min(target, self.max_position * float(risk["pl_gate_scale"]))

        delta = target - current_fraction
        skip_small_change = (
            not forced_reason
            and (
                self.days_since_order < int(risk["rebalance_days"])
                or abs(delta) < self.trade_threshold
            )
        )
        fee = 0.0
        closed_now = False
        traded = False
        was_long = self.shares > 0

        if not skip_small_change:
            target_value = target * portfolio
            trade_value = target_value - self.shares * price
            if trade_value > 1e-8:
                old_shares = self.shares
                execution_price = price * (1 + self.slippage)
                quantity = min(
                    trade_value / execution_price,
                    self.cash / (execution_price * (1 + self.commission)),
                )
                if quantity > 1e-10:
                    cost = quantity * execution_price
                    buy_fee = cost * self.commission
                    if old_shares <= 1e-10:
                        self.entry_value = 0.0
                        self.cycle_cost = 0.0
                        self.cycle_proceeds = 0.0
                        self.entry_date = self.view.dates[current_index]
                        self.peak_unrealized = 0.0
                        self.hold_steps = 0
                    self.cash -= cost + buy_fee
                    if abs(self.cash) < 1e-6:
                        self.cash = 0.0
                    self.shares += quantity
                    self.entry_value += cost + buy_fee
                    self.cycle_cost += cost + buy_fee
                    self.trade_count += 1
                    fee += buy_fee
                    traded = True
            elif trade_value < -1e-8:
                execution_price = price * (1 - self.slippage)
                quantity = min(-trade_value / execution_price, max(self.shares, 0.0))
                sell_fee, closed_now = self._sell(quantity, price, forced_reason or "policy")
                fee += sell_fee
                traded = quantity > 1e-10

        executed_fraction = self.shares * price / max(self.cash + self.shares * price, 1e-8)
        executed_delta = executed_fraction - current_fraction
        if traded:
            self.last_action = float(executed_fraction)
            self.days_since_order = 0
        else:
            self.last_action = float(current_fraction)
            self.days_since_order += 1
        self.total_fee += fee
        self.hold_steps = self.hold_steps + 1 if was_long and self.shares > 0 else (1 if self.shares > 0 else 0)

        # 使用下一交易日价格计算本步收益，确保奖励包含市场涨跌与全部交易成本。
        next_index = min(current_index + 1, len(self.view) - 1)
        next_price = float(self.view.close[next_index])
        before_nav = self.nav
        self.nav = max((self.cash + self.shares * next_price) / self.initial_cash, 1e-8)
        step_return = self.nav / max(before_nav, 1e-8) - 1
        self.return_history.append(float(step_return))
        self.position_fraction = float(np.clip(
            self.shares * next_price / max(self.cash + self.shares * next_price, 1e-8), 0.0, 1.0
        ))
        self.unrealized_pnl = (
            next_price / (self.entry_value / self.shares) - 1
            if self.shares > 0 and self.entry_value > 0 else 0.0
        )
        self.drawdown = float(1 - self.nav / max(self.peak_nav, self.nav))
        self.index = next_index
        self.done = bool(is_final)
        self.peak_nav = max(self.peak_nav, self.nav)
        self.drawdown = float(1 - self.nav / self.peak_nav)
        self.nav_history.append(float(self.nav))
        self.position_history.append(float(self.position_fraction))
        self.date_history.append(self.view.dates[self.index])
        self.price_history.append(next_price)

        recent = np.asarray(self.return_history[-int(cfg.get("reward_window", 60)):], dtype=np.float64)
        sharpe = 0.0
        calmar = 0.0
        if len(recent) >= int(cfg.get("min_reward_history", 20)):
            sharpe = float(recent.mean() / (recent.std() + 1e-8) * np.sqrt(242))
            recent_nav = np.asarray(self.nav_history[-(len(recent) + 1):], dtype=np.float64)
            recent_dd = float((1 - recent_nav / np.maximum.accumulate(recent_nav)).max())
            window_return = recent_nav[-1] / recent_nav[0] - 1
            annual = -1.0 if window_return <= -1 else (1 + window_return) ** (242 / len(recent)) - 1
            calmar = float(np.clip(annual, -1, 1) / (recent_dd + 0.05))

        reward = float(np.log(max(self.nav, 1e-8) / max(before_nav, 1e-8)) * 100)
        reward += cfg.get("sharpe_weight", 0.02) * np.clip(sharpe, -3, 3)
        reward += cfg.get("calmar_weight", 0.02) * np.clip(calmar, -3, 3)
        drawdown_scale = self._adaptive_drawdown_scale(high_volatility)
        drawdown_penalty = float(cfg.get("drawdown_weight", 5.0)) * drawdown_scale * self.drawdown ** 2
        min_closed = int(cfg.get("min_closed_trades", 3))
        if self.done and len(self.trade_returns) < min_closed:
            drawdown_penalty *= float(cfg.get("insufficient_closed_drawdown_multiplier", 1.5))
        reward -= drawdown_penalty
        reward -= cfg.get("turnover_weight", 0.2) * abs(executed_delta if traded else 0.0)
        if traded:
            reward -= cfg.get("trade_penalty", 0.02)
        if closed_now:
            reward += cfg.get("closed_trade_weight", 2.0) * np.clip(self.trade_returns[-1], -0.2, 0.2)
            profit_loss_min_trades = (
                int(cfg.get("profit_loss_min_trades_high_vol", 2)) if high_volatility
                else int(cfg.get("profit_loss_min_trades", 3))
            )
            if len(self.trade_returns) >= profit_loss_min_trades:
                reward += cfg.get("profit_loss_weight", 0.02) * np.clip(
                    np.log(max(self._profit_loss_ratio(), 0.05)), -2, 2
                )
            if self.losing_streak >= 3:
                reward -= cfg.get("consecutive_loss_penalty", 0.5) * (self.losing_streak - 2)

        trend_weight = float(cfg.get("trend_follow_weight", 0.02))
        if trend_active:
            holding_reward = trend_weight * abs(executed_fraction) * max(step_return * 100, 0)
            reward += holding_reward * float(cfg.get("trend_follow_multiplier", 1.2))
        elif downtrend_active and current_fraction > 0:
            holding_penalty = trend_weight * current_fraction * max(-step_return * 100, 1.0)
            reward -= holding_penalty * float(cfg.get("downtrend_holding_penalty_multiplier", 1.5))

        next_state = self._state() if not self.done else np.zeros_like(self.last_state)
        info = {
            "nav": self.nav, "sharpe": sharpe, "calmar": calmar,
            "profit_loss": self._profit_loss_ratio(), "drawdown_scale": drawdown_scale,
            "trend_active": bool(trend_active), "downtrend_active": bool(downtrend_active),
            "high_volatility": high_volatility, "forced_reason": forced_reason,
            "trade_count": self.trade_count, "closed_trades": len(self.trade_returns),
            "fee": float(fee),
        }
        return next_state, float(reward), self.done, info


class MultiStockTradingEnv:
    """每只股票独立状态与账户，训练器每步随机选择一只股票交互。"""

    def __init__(self, stocks, config, dataset="train", window=None):
        self.views = {code: stock.view(dataset, window) for code, stock in stocks.items()}
        self.envs = {code: TradingEnvV2(view, config) for code, view in self.views.items()}
        self.codes = list(self.envs)

    def reset(self):
        return {code: env.reset() for code, env in self.envs.items()}

    def active_codes(self):
        return [code for code in self.codes if not self.envs[code].done]

    def choose_code(self, rng):
        active = self.active_codes()
        return str(rng.choice(active)) if active else None

    def state(self, code):
        return self.envs[code].last_state

    def step(self, code, action):
        return self.envs[code].step(action)
