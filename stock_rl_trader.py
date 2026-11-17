# -*- coding: utf-8 -*-
"""
PPO 强化学习 A 股交易系统。

运行：
    pip install torch numpy pandas matplotlib tqdm akshare PyYAML
    python stock_rl_trader.py
    python stock_rl_trader.py --config config.yaml

设计约束：
- 默认使用 akshare 真实 A 股日线，东财/腾讯/新浪接口自动切换。
- 特征只使用历史行情，标准化统计量只由每只股票训练段计算。
- 每 50 个 episode 用最近训练窗口重新初始化环境；最后 20% 始终只做测试。
- 收盘后形成的状态在下一交易日开盘执行，收益按开盘价到下一开盘价计算。
- PPO 存储未裁剪动作及其原始 log_prob，环境执行时才裁剪，保证比率一致。
- 奖励包含收益、Sharpe、Calmar、盈亏比、回撤、换手和连续亏损惩罚。
- 仅研究用途，不构成投资建议；A 股默认仅做多/空仓，做空可在配置中开启。
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import akshare as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Normal

TRADING_DAYS = 242
FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "price_vs_ma5", "price_vs_ma20",
    "ma5_vs_ma20", "vol_ratio", "position_60", "vol_ratio_5", "ret_10",
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"seed", "data", "env", "ppo", "runtime"}
    missing = required - set(config)
    if missing:
        raise ValueError("配置缺少节点：%s" % sorted(missing))
    return config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simulate_daily(code, days=3000):
    seed = int(code) % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.018, days)
    close = 20 * np.cumprod(1 + returns)
    open_price = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, days))
    dates = pd.bdate_range("2010-01-01", periods=days)
    return pd.DataFrame({
        "date": dates, "open": open_price, "close": close,
        "volume": rng.lognormal(15, 0.5, days), "amount": np.nan,
    })


def fetch_akshare_daily(code, start, end, adjust, providers=None):
    """真实 A 股日线自动切换；不等待重试，不静默替换成模拟数据。"""
    adjust_text = "" if adjust == "none" else adjust
    symbol = ("sh" if code.startswith("6") else "sz") + code

    def eastmoney():
        func = getattr(ak, "stock_zh_a_hist_em", None) or getattr(ak, "stock_zh_a_hist", None)
        if func is None:
            raise RuntimeError("当前 akshare 缺少东方财富日线接口")
        return func(symbol=code, period="daily", start_date=start, end_date=end, adjust=adjust_text)

    def tencent():
        return ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust=adjust_text)

    def sina():
        return ak.stock_zh_a_daily_sina(symbol=symbol, start_date=start, end_date=end, adjust=adjust_text)

    available = {
        "eastmoney": ("东方财富", eastmoney),
        "tencent": ("腾讯", tencent),
        "sina": ("新浪", sina),
    }
    selected = providers or list(available)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError("未知行情接口：%s" % sorted(unknown))
    errors = []
    for provider in selected:
        source, loader = available[provider]
        try:
            raw = loader()
            if raw is None or raw.empty:
                raise RuntimeError("返回空数据")
            data = raw.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "成交量": "volume", "成交额": "amount",
            }).copy()
            for column in ["open", "close", "date"]:
                if column not in data:
                    raise RuntimeError("缺少字段 %s" % column)
            if "volume" not in data:
                data["volume"] = np.nan
            if "amount" not in data:
                data["amount"] = np.nan
            data = data[["date", "open", "close", "volume", "amount"]]
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            for column in ["open", "close", "volume", "amount"]:
                data[column] = pd.to_numeric(data[column], errors="coerce")
            data = data.dropna(subset=["date", "open", "close"])
            data = data.drop_duplicates("date", keep="last").sort_values("date")
            if data.empty or (data[["open", "close"]] <= 0).any().any():
                raise RuntimeError("无有效行情或存在非正价格，建议使用 hfq")
            print("  %s 使用 akshare/%s，%d 根日线" % (code, source, len(data)), flush=True)
            return data.reset_index(drop=True)
        except Exception as exc:
            errors.append("%s：%s" % (source, exc))
    raise RuntimeError("%s 所有真实行情接口均失败：\n%s" % (code, "\n".join(errors)))


class MarketView:
    """环境使用的轻量只读视图，特征固定为 float32，避免大额重复内存。"""

    def __init__(self, frame, features):
        self.dates = frame["date"].to_numpy()
        self.open = frame["open"].to_numpy(dtype=np.float64)
        self.close = frame["close"].to_numpy(dtype=np.float64)
        self.features = np.ascontiguousarray(features, dtype=np.float32)

    def __len__(self):
        return len(self.open)


class StockData:
    def __init__(self, code, data_config, test_ratio):
        self.code = code
        if data_config["source"] == "simulate":
            raw = simulate_daily(code, int(data_config.get("simulate_days", 3000)))
        else:
            end = data_config.get("end") or datetime.now().strftime("%Y%m%d")
            raw = fetch_akshare_daily(
                code, str(data_config["start"]), str(end), data_config.get("adjust", "hfq"),
                data_config.get("providers"),
            )
        frame = self._make_features(raw)
        self.split = int(len(frame) * (1 - test_ratio))
        validation_ratio = float(data_config.get("validation_ratio", 0.20))
        self.validation_split = int(self.split * (1 - validation_ratio))
        if self.validation_split < 300 or self.split - self.validation_split < 100 or len(frame) - self.split < 100:
            raise RuntimeError("%s 有效数据不足：训练/验证/测试段分别至少需 300/100/100 条" % code)
        raw_features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self.mean = raw_features[:self.validation_split].mean(axis=0)
        self.std = np.maximum(raw_features[:self.validation_split].std(axis=0), 1e-6)
        normalized = (raw_features - self.mean) / self.std
        frame = frame[["date", "open", "close"]].copy()
        self.frame = frame
        self.features = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _make_features(raw):
        data = raw.copy()
        close = data["close"]
        data["ret_1"] = close.pct_change(1)
        data["ret_3"] = close.pct_change(3)
        data["ret_5"] = close.pct_change(5)
        data["ret_10"] = close.pct_change(10)
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        data["price_vs_ma5"] = close / ma5 - 1
        data["price_vs_ma20"] = close / ma20 - 1
        data["ma5_vs_ma20"] = ma5 / ma20 - 1
        vol5 = data["ret_1"].rolling(5).std()
        vol20 = data["ret_1"].rolling(20).std()
        data["vol_ratio"] = vol5 / (vol20 + 1e-8)
        high60 = close.rolling(60).max()
        low60 = close.rolling(60).min()
        data["position_60"] = (close - low60) / (high60 - low60 + 1e-8)
        data["vol_ratio_5"] = data["volume"] / (data["volume"].rolling(5).mean() + 1e-8)
        data = data.replace([np.inf, -np.inf], np.nan)
        required = [column for column in FEATURE_COLUMNS if column != "vol_ratio_5"]
        data = data.dropna(subset=required + ["date", "open", "close"])
        data["vol_ratio_5"] = data["vol_ratio_5"].fillna(0.0)
        return data.reset_index(drop=True)

    def view(self, dataset="train", window=None):
        if dataset == "train":
            indexes = slice(0, self.validation_split)
        elif dataset == "validation":
            indexes = slice(self.validation_split, self.split)
        elif dataset == "test":
            indexes = slice(self.split, len(self.frame))
        else:
            raise ValueError("dataset 必须是 train、validation 或 test")
        frame = self.frame.iloc[indexes]
        features = self.features[indexes]
        if window and len(frame) > window:
            frame = frame.iloc[-window:]
            features = features[-window:]
        return MarketView(frame, features)


class TradingEnv:
    """净值法交易环境，避免原实现现金不足时仍强制设置目标持仓的问题。"""

    def __init__(self, view, config):
        self.view = view
        self.cfg = config
        self.lookback = int(config["lookback"])
        if len(view) <= self.lookback + 2:
            raise ValueError("环境数据不足，至少需要 lookback+2 根K线")
        self.reset()

    def reset(self, random_start=False, rng=None):
        self.step_idx = self.lookback
        if random_start and len(self.view) > self.lookback + 3:
            generator = rng or np.random.default_rng()
            self.step_idx = int(generator.integers(self.lookback, len(self.view) - 1))
        self.nav = 1.0
        self.position = 0.0
        self.max_nav = 1.0
        self.max_drawdown = 0.0
        self.done = False
        self.trade_count = 0
        self.hold_steps = 0
        self.losing_streak = 0
        self.trade_entry_nav = None
        self.trade_entry_date = None
        self.nav_history = [1.0]
        self.return_history = []
        self.position_history = [0.0]
        self.cost_history = [0.0]
        self.open_history = [self.view.open[self.step_idx]]
        self.date_history = [self.view.dates[self.step_idx]]
        self.trade_returns = []
        self.trade_log = []
        return self._get_state()

    def _get_state(self):
        window = self.view.features[self.step_idx - self.lookback:self.step_idx]
        recent = self.return_history[-5:]
        account = np.array([
            self.nav - 1.0,
            self.position,
            self.return_history[-1] if self.return_history else 0.0,
            (self.max_nav - self.nav) / (self.max_nav + 1e-8),
            np.mean(recent) if recent else 0.0,
            np.std(recent) if recent else 0.0,
            self.trade_count / 1000.0,
            self.hold_steps / 100.0,
            self.losing_streak / 10.0,
        ], dtype=np.float32)
        return np.concatenate([window.reshape(-1), account]).astype(np.float32)

    def _profit_loss_ratio(self):
        if not self.trade_returns:
            return 0.0
        returns = np.asarray(self.trade_returns)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) and len(losses):
            return float(np.clip(wins.mean() / abs(losses.mean()), 0, 5))
        return 5.0 if len(wins) else 0.0

    def _close_trade(self, exit_date):
        if self.trade_entry_nav is None:
            return False
        net_return = self.nav / self.trade_entry_nav - 1
        self.trade_returns.append(float(net_return))
        self.trade_log.append({
            "entry_date": str(self.trade_entry_date)[:10],
            "exit_date": str(exit_date)[:10],
            "net_return": float(net_return),
        })
        if net_return < 0:
            self.losing_streak += 1
        else:
            self.losing_streak = 0
        self.trade_entry_nav = None
        self.trade_entry_date = None
        return True

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, self._info(0.0, 0.0, 0.0)
        cfg = self.cfg
        current = self.step_idx
        next_index = current + 1
        old_position = self.position
        raw_action = float(np.asarray(action).reshape(-1)[0])
        lower = -float(cfg["max_position"]) if cfg.get("allow_short", False) else 0.0
        target = float(np.clip(raw_action, lower, float(cfg["max_position"])))
        threshold = float(cfg.get("trade_threshold", 0.01))
        if abs(target) < threshold:
            target = 0.0
        if abs(target - old_position) < threshold:
            target = old_position

        delta = target - old_position
        buy = max(delta, 0.0)
        sell = max(-delta, 0.0)
        cost = buy * (cfg["fee_rate"] + cfg["slip"])
        cost += sell * (cfg["fee_rate"] + cfg["slip"] + cfg["stamp_duty"])
        traded = abs(delta) >= threshold
        opens_trade = old_position == 0 and target != 0
        closes_trade = old_position != 0 and (target == 0 or old_position * target < 0)
        if opens_trade:
            self.trade_entry_nav = self.nav
            self.trade_entry_date = self.view.dates[current]

        before_nav = self.nav
        asset_return = self.view.open[next_index] / self.view.open[current] - 1
        step_return = target * asset_return - cost
        self.nav = max(0.0, self.nav * (1 + step_return))
        is_final_bar = next_index >= len(self.view) - 1
        if is_final_bar and cfg.get("liquidate_on_done", True) and target != 0:
            exit_stamp = cfg["stamp_duty"] if target > 0 else 0.0
            exit_cost = abs(target) * (cfg["fee_rate"] + cfg["slip"] + exit_stamp)
            self.nav *= max(0.0, 1 - exit_cost)
            cost += exit_cost
            target = 0.0
            closes_trade = True
            traded = True

        step_return = self.nav / before_nav - 1 if before_nav > 0 else 0.0
        self.position = target
        if traded:
            self.trade_count += 1
            self.hold_steps = 0
        else:
            self.hold_steps += 1
        self.return_history.append(float(step_return))
        self.nav_history.append(float(self.nav))
        self.position_history.append(float(target))
        self.cost_history.append(float(cost))
        self.open_history.append(self.view.open[next_index])
        self.date_history.append(self.view.dates[next_index])
        self.max_nav = max(self.max_nav, self.nav)
        drawdown = (self.max_nav - self.nav) / (self.max_nav + 1e-8)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        closed_now = self._close_trade(self.view.dates[next_index]) if closes_trade else False
        if closes_trade and target != 0:
            self.trade_entry_nav = self.nav
            self.trade_entry_date = self.view.dates[next_index]

        recent = np.asarray(self.return_history[-int(cfg["reward_window"]):], dtype=np.float64)
        min_history = int(cfg.get("min_reward_history", 20))
        sharpe = 0.0
        calmar = 0.0
        if len(recent) >= min_history:
            sharpe = float(recent.mean() / (recent.std() + 1e-8) * np.sqrt(TRADING_DAYS))
            recent_nav = np.asarray(self.nav_history[-(len(recent) + 1):], dtype=np.float64)
            recent_dd = float((1 - recent_nav / np.maximum.accumulate(recent_nav)).max())
            window_return = recent_nav[-1] / recent_nav[0] - 1
            annual_return = -1.0 if window_return <= -1 else (1 + window_return) ** (TRADING_DAYS / len(recent)) - 1
            # 限制短期年化极端值，并给回撤设置 5% 下限，避免早期 Calmar 爆炸。
            calmar = float(np.clip(annual_return, -1, 1) / (recent_dd + 0.05))
        profit_loss = self._profit_loss_ratio()

        # 以对数收益为主，风险比率只在足够历史后生效，降低小样本噪声和过拟合。
        reward = float(np.log(max(self.nav, 1e-8) / max(before_nav, 1e-8)) * 100)
        reward += cfg["sharpe_weight"] * np.clip(sharpe, -3, 3)
        reward += cfg["calmar_weight"] * np.clip(calmar, -3, 3)
        reward -= cfg["drawdown_weight"] * drawdown ** 2
        reward -= cfg["turnover_weight"] * abs(delta)
        if traded:
            reward -= cfg["trade_penalty"]
        if closed_now:
            trade_return = self.trade_returns[-1]
            reward += cfg.get("closed_trade_weight", 0.0) * np.clip(trade_return, -0.2, 0.2)
            # 至少 5 笔交易后才估计盈亏比；以 1 为中心，差盈亏比也会受罚。
            if len(self.trade_returns) >= 5:
                reward += cfg["profit_loss_weight"] * np.clip(np.log(max(profit_loss, 0.05)), -2, 2)
            if self.losing_streak >= 3:
                reward -= cfg["consecutive_loss_penalty"] * (self.losing_streak - 2)
        if self.hold_steps > 5:
            reward += cfg.get("hold_reward", 0.0)

        bankrupt = self.nav <= 0.3 or step_return <= -0.999
        if bankrupt:
            reward -= 100
        self.step_idx = next_index
        self.done = bool(is_final_bar or bankrupt)
        info = self._info(sharpe, calmar, profit_loss, cost)
        return self._get_state(), float(reward), self.done, info

    def _info(self, sharpe, calmar, profit_loss, cost):
        return {
            "nav": self.nav, "drawdown": self.max_drawdown, "sharpe": sharpe,
            "calmar": calmar, "profit_loss": profit_loss, "position": self.position,
            "trade_count": self.trade_count, "closed_trades": len(self.trade_returns),
            "losing_streak": self.losing_streak, "cost": cost,
        }

    def frame(self):
        benchmark = np.asarray(self.open_history) / self.open_history[0]
        return pd.DataFrame({
            "date": pd.to_datetime(self.date_history), "strategy_nav": self.nav_history,
            "benchmark_nav": benchmark, "position": self.position_history,
            "cost_rate": self.cost_history,
            "strategy_return": np.r_[0.0, self.return_history],
        })


def performance_metrics(env):
    returns = np.asarray(env.return_history, dtype=np.float64)
    final_nav = float(env.nav)
    years = max(len(returns) / TRADING_DAYS, 1 / TRADING_DAYS)
    annual = final_nav ** (1 / years) - 1 if final_nav > 0 else -1.0
    sharpe = float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(TRADING_DAYS)) if len(returns) else 0.0
    trades = np.asarray(env.trade_returns, dtype=np.float64)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    if len(wins) and len(losses):
        profit_loss = float(wins.mean() / abs(losses.mean()))
    elif len(wins):
        profit_loss = float("inf")
    else:
        profit_loss = 0.0
    return {
        "final_nav": final_nav,
        "total_return": final_nav - 1,
        "annual_return": float(annual),
        "sharpe": sharpe,
        "max_drawdown": float(env.max_drawdown),
        "calmar": float(annual / (env.max_drawdown + 1e-8)),
        "win_rate": float((trades > 0).mean()) if len(trades) else 0.0,
        "profit_loss": profit_loss,
        "trade_count": int(env.trade_count),
        "closed_trades": int(len(trades)),
        "losing_streak": int(env.losing_streak),
    }


class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1))
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor_mean(features)
        std = torch.exp(torch.clamp(self.actor_logstd, -5, 2))
        value = self.critic(features).squeeze(-1)
        return mean, std, value


class PPOAgent:
    def __init__(self, state_dim, config, device):
        self.cfg = config
        self.device = device
        self.policy = ActorCritic(state_dim, int(config["hidden_dim"])).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=float(config["lr"]), eps=1e-5)
        self.memory = {key: [] for key in ["states", "actions", "log_probs", "rewards", "values", "dones"]}

    def select_action(self, state, deterministic=False):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.policy(tensor)
            distribution = Normal(mean, std)
            action = mean if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
        return float(action.item()), float(log_prob.item()), float(value.item())

    def value(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.policy(tensor)[2].item())

    def store(self, state, action, log_prob, reward, value, done):
        self.memory["states"].append(np.asarray(state, dtype=np.float32))
        self.memory["actions"].append(float(action))
        self.memory["log_probs"].append(float(log_prob))
        self.memory["rewards"].append(float(reward))
        self.memory["values"].append(float(value))
        self.memory["dones"].append(bool(done))

    def update(self, last_value):
        count = len(self.memory["states"])
        if count < 2:
            self.memory = {key: [] for key in self.memory}
            return {}
        cfg = self.cfg
        rewards = self.memory["rewards"]
        values = self.memory["values"]
        dones = self.memory["dones"]
        advantages = np.zeros(count, dtype=np.float32)
        last_advantage = 0.0
        for index in reversed(range(count)):
            next_value = last_value if index == count - 1 else values[index + 1]
            nonterminal = 1.0 - float(dones[index])
            delta = rewards[index] + cfg["gamma"] * next_value * nonterminal - values[index]
            last_advantage = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterminal * last_advantage
            advantages[index] = last_advantage
        returns = advantages + np.asarray(values, dtype=np.float32)

        states = torch.as_tensor(np.asarray(self.memory["states"], dtype=np.float32), device=self.device)
        actions = torch.as_tensor(self.memory["actions"], dtype=torch.float32, device=self.device).unsqueeze(1)
        old_log_probs = torch.as_tensor(self.memory["log_probs"], dtype=torch.float32, device=self.device)
        advantage_tensor = torch.as_tensor(advantages, device=self.device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        self.policy.train()
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        batches = 0
        for _ in range(int(cfg["epochs_per_update"])):
            indexes = np.random.permutation(count)
            for start in range(0, count, int(cfg["batch_size"])):
                selected = torch.as_tensor(
                    indexes[start:start + int(cfg["batch_size"])], dtype=torch.long, device=self.device
                )
                mean, std, values_now = self.policy(states[selected])
                distribution = Normal(mean, std)
                log_probs = distribution.log_prob(actions[selected]).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1).mean()
                ratio = torch.exp(log_probs - old_log_probs[selected])
                batch_advantage = advantage_tensor[selected]
                surrogate1 = ratio * batch_advantage
                surrogate2 = torch.clamp(
                    ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]
                ) * batch_advantage
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = nn.functional.mse_loss(values_now, return_tensor[selected])
                loss = (
                    policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach())
                totals["value_loss"] += float(value_loss.detach())
                totals["entropy"] += float(entropy.detach())
                batches += 1
        self.memory = {key: [] for key in self.memory}
        return {key: value / max(batches, 1) for key, value in totals.items()}


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 无法使用 GPU")
    return torch.device(name)


def save_checkpoint(path, agent, episode, config, state_dim):
    torch.save({
        "episode": episode,
        "model_state_dict": {key: value.detach().cpu() for key, value in agent.policy.state_dict().items()},
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "state_dim": state_dim,
        "feature_columns": FEATURE_COLUMNS,
        "config": config,
    }, path)


def train_agent(primary, config, device, output_dir):
    env_config = config["env"]
    ppo_config = config["ppo"]
    rolling_window = int(config["data"].get("rolling_window", 1500))
    view = primary.view("train", rolling_window)
    env = TradingEnv(view, env_config)
    state_dim = len(env.reset())
    agent = PPOAgent(state_dim, ppo_config, device)
    rng = np.random.default_rng(config["seed"])
    best_score = -np.inf
    best_state = None
    log_path = output_dir / "training_log.csv"
    fields = [
        "episode", "reward", "nav", "annual_return", "sharpe", "max_drawdown",
        "calmar", "win_rate", "profit_loss", "trade_count", "closed_trades",
        "losing_streak", "policy_loss", "value_loss", "entropy",
        "validation_score", "validation_nav", "validation_sharpe", "validation_drawdown",
        "window_start", "window_end",
    ]
    print("训练股票：%s，状态维度：%d，设备：%s" % (primary.code, state_dim, device), flush=True)
    with log_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        total_episodes = int(ppo_config["total_episodes"])
        validation_every = int(ppo_config.get("validation_every", 10))
        for episode in range(1, total_episodes + 1):
            if episode == 1 or (episode - 1) % int(ppo_config["reinit_every"]) == 0:
                view = primary.view("train", rolling_window)
                env = TradingEnv(view, env_config)
                print(
                    "重新初始化环境：episode %d，%s ~ %s，%d 根K线"
                    % (episode, str(view.dates[0])[:10], str(view.dates[-1])[:10], len(view)),
                    flush=True,
                )
            state = env.reset(random_start=bool(ppo_config.get("random_start", True)), rng=rng)
            done = False
            episode_reward = 0.0
            steps = 0
            while not done and steps < int(ppo_config["rollout_steps"]):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store(state, action, log_prob, reward, value, done)
                state = next_state
                episode_reward += reward
                steps += 1
            last_value = 0.0 if done else agent.value(state)
            losses = agent.update(last_value)
            summary = performance_metrics(env)

            should_validate = episode % validation_every == 0 or episode == total_episodes
            validation_score = ""
            validation_summary = None
            if should_validate:
                agent.policy.eval()
                _, validation_summary = evaluate(agent, primary, "validation", config)
                validation_score = float(
                    validation_summary["sharpe"]
                    + 0.5 * np.clip(validation_summary["calmar"], -5, 5)
                    - 0.5 * validation_summary["max_drawdown"]
                )
                agent.policy.train()

            row = {
                "episode": episode, "reward": episode_reward,
                **{key: summary[key] for key in [
                    "annual_return", "sharpe", "max_drawdown", "calmar", "win_rate",
                    "profit_loss", "trade_count", "closed_trades", "losing_streak",
                ]},
                "nav": summary["final_nav"],
                "policy_loss": losses.get("policy_loss", 0.0),
                "value_loss": losses.get("value_loss", 0.0),
                "entropy": losses.get("entropy", 0.0),
                "validation_score": validation_score,
                "validation_nav": validation_summary["final_nav"] if validation_summary else "",
                "validation_sharpe": validation_summary["sharpe"] if validation_summary else "",
                "validation_drawdown": validation_summary["max_drawdown"] if validation_summary else "",
                "window_start": str(view.dates[0])[:10], "window_end": str(view.dates[-1])[:10],
            }
            writer.writerow(row)
            handle.flush()
            if episode % int(ppo_config.get("print_every", 1)) == 0:
                print(
                    "Episode %4d | 奖励 %8.3f | 净值 %.4f | 回撤 %.2f%% | "
                    "夏普 %.2f | Calmar %.2f | 盈亏比 %.2f | 交易 %d/%d"
                    % (
                        episode, episode_reward, summary["final_nav"], summary["max_drawdown"] * 100,
                        summary["sharpe"], summary["calmar"], summary["profit_loss"],
                        summary["trade_count"], summary["closed_trades"],
                    ), flush=True,
                )
            if should_validate:
                print(
                    "  验证 | 得分 %.3f | 净值 %.4f | 夏普 %.2f | 回撤 %.2f%%"
                    % (
                        validation_score, validation_summary["final_nav"],
                        validation_summary["sharpe"], validation_summary["max_drawdown"] * 100,
                    ), flush=True,
                )
                if validation_score > best_score:
                    best_score = validation_score
                    best_state = {
                        key: value.detach().cpu().clone() for key, value in agent.policy.state_dict().items()
                    }
                    save_checkpoint(output_dir / "model_best.pth", agent, episode, config, state_dim)
    save_checkpoint(output_dir / "model_final.pth", agent, total_episodes, config, state_dim)
    if best_state is not None:
        agent.policy.load_state_dict(best_state)
    agent.policy.eval()
    return agent


def evaluate(agent, stock, dataset, config):
    env = TradingEnv(stock.view(dataset), config["env"])
    state = env.reset()
    done = False
    while not done:
        action, _, _ = agent.select_action(state, deterministic=True)
        state, _, done, _ = env.step(action)
    return env, performance_metrics(env)


def save_training_curves(log_path, output_path):
    log = pd.read_csv(log_path)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, column, title in [
        (axes[0, 0], "reward", "Episode Reward"),
        (axes[0, 1], "nav", "Final NAV"),
        (axes[1, 0], "max_drawdown", "Max Drawdown"),
        (axes[1, 1], "sharpe", "Sharpe"),
    ]:
        axis.plot(log["episode"], log[column])
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_reports(agent, stocks, primary_code, config, output_dir):
    rows = []
    frames = {}
    for code, stock in stocks.items():
        for dataset in ["train", "validation", "test"]:
            env, metrics = evaluate(agent, stock, dataset, config)
            rows.append({"code": code, "dataset": dataset, **metrics})
            frame = env.frame()
            frames[(code, dataset)] = frame
            frame.to_csv(
                output_dir / ("nav_%s_%s.csv" % (code, dataset)),
                index=False, encoding="utf-8-sig", date_format="%Y-%m-%d",
            )
            pd.DataFrame(env.trade_log).to_csv(
                output_dir / ("trades_%s_%s.csv" % (code, dataset)),
                index=False, encoding="utf-8-sig",
            )
            print(
                "%s %s | 净值 %.4f | 年化 %.2f%% | 夏普 %.2f | 回撤 %.2f%% | "
                "Calmar %.2f | 胜率 %.2f%% | 盈亏比 %.2f | 交易 %d/%d"
                % (
                    code, dataset, metrics["final_nav"], metrics["annual_return"] * 100,
                    metrics["sharpe"], metrics["max_drawdown"] * 100, metrics["calmar"],
                    metrics["win_rate"] * 100, metrics["profit_loss"], metrics["trade_count"],
                    metrics["closed_trades"],
                ), flush=True,
            )
    pd.DataFrame(rows).to_csv(output_dir / "backtest_report.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    for axis, dataset in zip(axes, ["train", "test"]):
        frame = frames[(primary_code, dataset)]
        axis.plot(frame["date"], frame["strategy_nav"], label="PPO 策略")
        axis.plot(frame["date"], frame["benchmark_nav"], "--", label="买入持有")
        axis.set_title("%s - %s" % (primary_code, "训练集" if dataset == "train" else "纯样本外测试集"))
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "rl_nav.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    for code in stocks:
        frame = frames[(code, "test")]
        axis.plot(frame["date"], frame["strategy_nav"], label=code)
    axis.set_title("多股票纯样本外净值比较")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "multi_stock_test_nav.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="PPO 强化学习 A 股交易系统")
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config.yaml")), help="YAML 配置文件路径"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    runtime = config["runtime"]
    torch.set_num_threads(int(runtime.get("cpu_threads", 4)))
    device = resolve_device(runtime.get("device", "auto"))
    output_dir = Path(runtime.get("output_dir", "rl_output"))
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    print("=" * 72, flush=True)
    print("PPO 强化学习股票交易系统，仅研究用途，不构成投资建议", flush=True)
    print("=" * 72, flush=True)
    stocks = {}
    for code in config["data"]["codes"]:
        code = str(code).zfill(6)
        stocks[code] = StockData(code, config["data"], float(config["data"]["test_ratio"]))
    primary_code = str(config["data"].get("train_code") or config["data"]["codes"][0]).zfill(6)
    if primary_code not in stocks:
        raise ValueError("train_code 必须包含在 codes 中")
    agent = train_agent(stocks[primary_code], config, device, output_dir)
    save_training_curves(output_dir / "training_log.csv", output_dir / "training_curves.png")
    save_reports(agent, stocks, primary_code, config, output_dir)
    metadata = {
        "primary_code": primary_code,
        "device": str(device),
        "disclaimer": "仅研究用途，不构成投资建议",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("模型、日志和完整报告已保存到：%s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
