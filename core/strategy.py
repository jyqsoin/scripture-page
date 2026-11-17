# -*- coding: utf-8 -*-
"""
backtrader A 股日线策略：MA 金叉买入 + 固定止损 / ATR 追踪止盈止损。

风控说明（V1.1 Step F2）：
- 默认 enable_atr=False，完全沿用 V1.0 的固定百分比止损（买入价下方 stop_loss）；
- 启用 ATR 后同时存在两道防线：
  1) 初始 ATR 止损：entry - atr_stop_multiplier * ATR(成交当日)；
  2) ATR 追踪止盈：peak(持仓最高价) - atr_trailing_multiplier * ATR(最新)；
- 固定止损始终保留，与 ATR 止损取更严格者（更早触发），作为底线；
- ATR 在持仓期间每日重算，不在买入时一次性缓存，避免阈值滞后。

交易约束：
- 收盘确认：T 日收盘生成信号，T+1 日开盘成交，不在 T 日收盘作弊成交；
- 必须使用 cerebro.broker.set_coo(True)，策略在 next_open 中处理次日开盘订单；
- T+1：成交当日不允许卖出；
- 开盘涨停不买、开盘跌停不卖；无当日 K 线视为停牌，禁止开仓；
- 单票市值不超过组合 10%，总股票市值不超过组合 80%；
- 买入数量为 100 股整数倍，现金预留手续费和滑点；
- next_open 中只使用 open[0] 和 close[-1]，绝不使用次日 close[0]，避免未来函数。

价格口径说明：V1.0 用同一复权序列的前收盘近似计算涨跌停价；除权日可能与交易所
真实涨跌停价有差异，后续可用不复权行情额外提供真实涨跌停价。

仅研究用途，不构成投资建议。
"""

import math

import backtrader as bt


class MACrossStrategy(bt.Strategy):
    """多股票 MA 金叉策略；只做多，不做空，不自动加仓。"""

    params = dict(
        fast_period=5,
        slow_period=20,
        stop_loss=0.05,
        # ---- V1.1 Step F2：ATR 追踪止盈止损（默认禁用，保持 V1.0 行为） ----
        enable_atr=False,          # 是否启用 ATR 风控
        atr_period=14,             # ATR 计算周期
        atr_stop_multiplier=2.0,   # 初始 ATR 止损倍数：entry - N*ATR
        atr_trailing_multiplier=3.0,  # 追踪止盈倍数：peak - N*ATR(最新)
        # ------------------------------------------------------------------
        max_position_per_stock=0.10,
        max_total_position=0.80,
        commission=0.0003,
        slippage=0.001,
        lot_size=100,
        fixed_amount=None,       # 指定后按固定金额买入，仍受单票/总仓上限约束
        print_log=False,
    )

    def __init__(self):
        self.ma_fast = {}
        self.ma_slow = {}
        self.crossover = {}
        self.atr = {}
        for data in self.datas:
            self.ma_fast[data] = bt.ind.SMA(data.close, period=self.p.fast_period)
            self.ma_slow[data] = bt.ind.SMA(data.close, period=self.p.slow_period)
            self.crossover[data] = bt.ind.CrossOver(self.ma_fast[data], self.ma_slow[data])
            if self.p.enable_atr:
                self.atr[data] = bt.ind.ATR(data, period=self.p.atr_period)

        # 上一收盘产生、等待下一开盘执行的信号。
        self.pending_buy = set()
        self.pending_sell = set()
        self.forced_exit = set()  # 止损一旦触发，跌停/停牌后也继续等待可卖时机
        self.active_orders = {data: None for data in self.datas}
        self.buy_price = {}
        self.entry_date = {}
        # ATR 风控状态：peak 跟踪持仓最高价，atr_at_entry 记录成交当日 ATR。
        self.peak_price = {}
        self.atr_at_entry = {}
        self.exit_reason = {}      # 记录每只持仓的最新触发原因，供卖出事件使用
        self.events = []          # 给回测明细和 UI 使用
        self.warnings = []        # 拒单、废单及规则拦截记录

    @staticmethod
    def _round_cent(price):
        """A 股价格最小单位 0.01 元，按四舍五入（half-up）处理。"""
        return math.floor(float(price) * 100 + 0.5) / 100.0

    @staticmethod
    def _code_of(data):
        return getattr(data, "_name", "") or "UNKNOWN"

    def _limit_rate(self, data):
        """创业板/科创板 20%，其余沪深主板 10%；股票池默认已排除 ST。"""
        code = self._code_of(data)
        if code.startswith(("30", "68")):
            return 0.20
        return 0.10

    def _limit_prices(self, data):
        """次日开盘只能参考前收盘计算当日涨跌停价。"""
        previous_close = float(data.close[-1])
        rate = self._limit_rate(data)
        return (
            self._round_cent(previous_close * (1 + rate)),
            self._round_cent(previous_close * (1 - rate)),
        )

    def _current_data_date(self):
        """next_open 中策略时钟尚未前进，需使用各数据源当前周期的最新日期。"""
        return max(item.datetime.date(0) for item in self.datas if len(item))

    def _is_trading_today(self, data):
        """数据日期与当前数据时钟一致才表示当日有行情，否则按停牌处理。"""
        return data.datetime.date(0) == self._current_data_date()

    def _record(self, data, action, price=0.0, size=0, reason=""):
        event = {
            "date": data.datetime.date(0).strftime("%Y-%m-%d"),
            "code": self._code_of(data),
            "action": action,
            "price": float(price or 0),
            "size": int(size or 0),
            "reason": reason,
        }
        self.events.append(event)
        if self.p.print_log:
            print(event)

    def _block(self, data, action, reason):
        self.warnings.append({
            "date": data.datetime.date(0).strftime("%Y-%m-%d"),
            "code": self._code_of(data), "action": action, "reason": reason,
        })

    def _safe_atr(self, data):
        """取当前 ATR，NaN/0/负数时返回 None，避免风控阈值失效。"""
        atr_value = self.atr.get(data)
        if atr_value is None:
            return None
        try:
            value = float(atr_value[0])
        except (TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    def _check_exit(self, data, cost):
        """
        判断持仓是否触发卖出。

        返回 (是否触发, 触发原因)。多道防线取最早触发者：
        - 固定止损（始终生效）：close <= entry * (1 - stop_loss)
        - ATR 初始止损（启用）：close <= entry - atr_stop * ATR(成交日)
        - ATR 追踪止盈（启用）：close <= peak - atr_trail * ATR(最新)
        """
        close = float(data.close[0])

        # 1) 固定百分比止损：V1.0 底线，始终保留。
        fixed_stop = cost * (1 - self.p.stop_loss)
        if close <= fixed_stop:
            return True, "固定止损 %.2f%%" % (self.p.stop_loss * 100)

        if not self.p.enable_atr:
            return False, ""

        # 2) ATR 初始止损：基准为买入价，ATR 取成交当日值。
        atr_entry = self.atr_at_entry.get(data)
        if atr_entry:
            atr_stop_price = cost - self.p.atr_stop_multiplier * atr_entry
            # 仅在 ATR 止损比固定止损更严格时才使用，避免重复触发同一笔。
            if atr_stop_price > fixed_stop and close <= atr_stop_price:
                return True, "ATR 止损(%.1f×ATR)" % self.p.atr_stop_multiplier

        # 3) ATR 追踪止盈：基准为持仓最高价，ATR 取最新值（每日重算）。
        current_atr = self._safe_atr(data)
        peak = self.peak_price.get(data, cost)
        if current_atr and peak > cost:
            trail_stop = peak - self.p.atr_trailing_multiplier * current_atr
            if close <= trail_stop:
                return True, "ATR 追踪止盈(%.1f×ATR)" % self.p.atr_trailing_multiplier

        return False, ""

    def next(self):
        """
        收盘确认阶段：只根据已经完成的当日 K 线生成信号，不在此直接下单。

        Market 订单留到 next_open，从而避免信号日收盘成交，也便于次日开盘过滤涨跌停。
        """
        self.pending_buy = set()
        self.pending_sell = set()
        for data in self.datas:
            if not self._is_trading_today(data):
                continue
            position = self.getposition(data)
            if position.size > 0:
                cost = self.buy_price.get(data)
                if cost:
                    # 持仓期间持续更新最高价，作为追踪止盈基准。
                    self.peak_price[data] = max(
                        self.peak_price.get(data, cost), float(data.high[0])
                    )
                    triggered, reason = self._check_exit(data, cost)
                    if data in self.forced_exit or triggered:
                        # 从止损首次触发起持续保留，直到真正清仓，避免拒单/跌停后丢失信号。
                        self.forced_exit.add(data)
                        self.pending_sell.add(data)
                        if triggered and reason:
                            self.exit_reason[data] = reason
            elif position.size == 0 and self.active_orders[data] is None:
                if self.crossover[data][0] > 0:
                    self.pending_buy.add(data)

    def _price_at_open(self, data):
        """组合估值只取开盘价；停牌股票用其最后可见收盘价。"""
        return float(data.open[0]) if self._is_trading_today(data) else float(data.close[0])

    def _portfolio_at_open(self):
        cash = float(self.broker.getcash())
        invested = 0.0
        for data in self.datas:
            position = self.getposition(data)
            if position.size:
                invested += position.size * self._price_at_open(data)
        return cash, invested, cash + invested

    def next_open(self):
        """次日开盘执行上一收盘信号；cheat-on-open 下 Market 订单按当根开盘撮合。"""
        if not self.pending_buy and not self.pending_sell:
            return
        today = self._current_data_date()
        cash, invested, equity = self._portfolio_at_open()
        planned_sale_value = 0.0

        # 先处理卖出；卖出所得不用于同日买入，现金口径保持保守。
        for data in list(self.pending_sell):
            position = self.getposition(data)
            if position.size <= 0 or self.active_orders[data] is not None:
                continue
            if not self._is_trading_today(data):
                self.forced_exit.add(data)
                self._block(data, "SELL_BLOCKED", "停牌，无法卖出，保留止损状态")
                continue
            entry = self.entry_date.get(data)
            if entry is not None and today <= entry:
                self.forced_exit.add(data)
                self._block(data, "SELL_BLOCKED", "T+1，买入当日不可卖")
                continue
            open_price = float(data.open[0])
            _, limit_down = self._limit_prices(data)
            if open_price <= limit_down + 1e-8:
                self.forced_exit.add(data)
                self._block(data, "SELL_BLOCKED", "开盘跌停，暂不卖，保留止损状态")
                continue
            order = self.sell(data=data, size=position.size, exectype=bt.Order.Market)
            self.active_orders[data] = order
            planned_sale_value += position.size * open_price
            reason = self.exit_reason.get(data, "风控触发")
            self._record(data, "SELL_SIGNAL", open_price, position.size, reason)

        planned_invested = max(0.0, invested - planned_sale_value)
        reserved_cash = 0.0
        reserved_invested = 0.0

        for data in list(self.pending_buy):
            if self.getposition(data).size != 0 or self.active_orders[data] is not None:
                continue
            if not self._is_trading_today(data):
                self._block(data, "BUY_BLOCKED", "停牌，禁止开仓")
                continue
            open_price = float(data.open[0])
            if open_price <= 0:
                self._block(data, "BUY_BLOCKED", "开盘价非法")
                continue
            limit_up, _ = self._limit_prices(data)
            if open_price >= limit_up - 1e-8:
                self._block(data, "BUY_BLOCKED", "开盘涨停，不买")
                continue

            # 固定金额或组合百分比目标，两者都不能突破单票和总仓红线。
            target = (float(self.p.fixed_amount) if self.p.fixed_amount
                      else equity * self.p.max_position_per_stock)
            target = min(target, equity * self.p.max_position_per_stock)
            total_budget = (equity * self.p.max_total_position - planned_invested
                            - reserved_invested)
            cash_budget = (cash - reserved_cash) / (1 + self.p.commission + self.p.slippage)
            budget = min(target, total_budget, cash_budget)
            size = int(budget / open_price / self.p.lot_size) * self.p.lot_size
            if size <= 0:
                self._block(data, "BUY_BLOCKED", "仓位或现金不足一手")
                continue

            order = self.buy(data=data, size=size, exectype=bt.Order.Market)
            self.active_orders[data] = order
            estimated_cost = size * open_price * (1 + self.p.commission + self.p.slippage)
            reserved_cash += estimated_cost
            reserved_invested += size * open_price
            self._record(data, "BUY_SIGNAL", open_price, size, "MA 金叉后次日开盘买入")

        self.pending_buy.clear()
        self.pending_sell.clear()

    def notify_order(self, order):
        """记录真实成交与拒单；买入成交价用于后续止损，ATR 状态同步初始化/清理。"""
        data = order.data
        if order.status in (order.Submitted, order.Accepted):
            self.active_orders[data] = order
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.forced_exit.discard(data)
                self.buy_price[data] = float(order.executed.price)
                self.entry_date[data] = data.datetime.date(0)
                # 成交当日初始化 ATR 风控状态。
                self.peak_price[data] = float(order.executed.price)
                self.atr_at_entry[data] = self._safe_atr(data)
                self.exit_reason.pop(data, None)
                self._record(data, "BUY", order.executed.price, order.executed.size, "开盘成交")
            elif order.issell():
                self._record(data, "SELL", order.executed.price, order.executed.size, "开盘成交")
                if self.getposition(data).size == 0:
                    self.forced_exit.discard(data)
                    self.buy_price.pop(data, None)
                    self.entry_date.pop(data, None)
                    self.peak_price.pop(data, None)
                    self.atr_at_entry.pop(data, None)
                    self.exit_reason.pop(data, None)
        elif order.status in (order.Canceled, order.Margin, order.Rejected, order.Expired):
            self._block(data, "ORDER_FAILED", "订单状态：%s" % order.getstatusname())
        self.active_orders[data] = None


if __name__ == "__main__":
    # 合成行情最小回测：只验证策略可实例化和运行，不验证任何收益。
    import numpy as np
    import pandas as pd

    print("仅研究用途，不构成投资建议")
    size = 100
    index = pd.bdate_range("2024-01-01", periods=size)
    close = pd.Series(10 + np.sin(np.arange(size) / 4) * 2 + np.arange(size) * 0.02)
    frame = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]).to_numpy(),
        "high": (close + 0.2).to_numpy(),
        "low": (close - 0.2).to_numpy(),
        "close": close.to_numpy(),
        "volume": np.full(size, 1000000.0),
        "openinterest": np.zeros(size),
    }, index=index)

    cerebro = bt.Cerebro()
    cerebro.broker.set_coo(True)  # 收盘信号次日开盘成交的必要开关
    cerebro.broker.setcash(1000000)
    cerebro.broker.setcommission(commission=0.0003)
    feed = bt.feeds.PandasData(dataname=frame)
    cerebro.adddata(feed, name="600000")
    cerebro.addstrategy(MACrossStrategy, fast_period=3, slow_period=8, print_log=False)
    strategies = cerebro.run()
    strategy = strategies[0]
    print("期末权益：%.2f" % cerebro.broker.getvalue())
    print("策略事件数：", len(strategy.events))
    print("最近事件：", strategy.events[-5:])
    print("strategy.py 自测运行完成；本结果仅研究用途")
