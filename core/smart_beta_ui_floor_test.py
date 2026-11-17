# -*- coding: utf-8 -*-
"""I2-alt-12 Smart Beta UI 动量最低权重接入自测。

覆盖三件事：
1. 合成行情验证 run_fundamental_comparison 只把 momentum_floor 传给五因子，
   四因子及双/三因子结果逐值不变；
2. 五因子并集分支在启用 10% 下限后，动量权重确实达到下限；
3. 离屏 Qt 实例化 SmartBetaPage，验证控件默认 10% 且仅 fundamental 方案传参。

不修改核心默认值：不传 momentum_floor 时等价于旧规则。
仅研究用途，不构成投资建议。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import smart_beta_fundamental_backtest as sbf


def _synthetic_market():
    """构造五组正交排序的合成行情，确保并集分支且动量股不在前 30 名次和。"""
    dates = pd.bdate_range("2022-12-01", "2024-08-30").strftime("%Y-%m-%d").tolist()
    month_end = pd.DataFrame({"date": dates,
                              "period": pd.to_datetime(dates).to_period("M")})
    rebalance = month_end.groupby("period")["date"].last().tolist()
    rebalance = [date for date in rebalance if date >= "2024-06-01"]
    assert len(rebalance) >= 3, "合成调仓日不足"

    rng = np.random.default_rng(20261115)
    data_map, valuation_map = {}, {}
    for j in range(50):
        code = "%06d" % (j + 1)
        group = j % 5
        volatility = 0.003 if group == 1 else (0.04 if group == 2 else 0.02)
        drift = 0.004 if group == 2 else 0.0
        returns = drift + rng.normal(0, volatility, len(dates))
        if group == 0:
            start = 1.0 + 0.1 * (j // 5)          # 低价组
        elif group == 2:
            start = 200.0 + j                     # 动量组价格最高
        else:
            start = 100.0 + j
        data_map[code] = pd.DataFrame(
            {"date": dates, "close": start * np.cumprod(1 + returns)})

        if group == 3:
            pe = 3.0 + 0.1 * (j // 5)             # 低 PE 组
        elif group == 2:
            pe = 200.0 + j
        else:
            pe = 50.0 + j
        if group == 4:
            pb = 0.3 + 0.01 * (j // 5)            # 低 PB 组
        elif group == 2:
            pb = 10.0 + j
        else:
            pb = 3.0 + 0.01 * j
        valuation_map[code] = pd.DataFrame({
            "date": [dates[0], dates[-1]], "code": [code, code],
            "pe_ttm": [pe, pe], "pb_mrq": [pb, pb]})
    return data_map, valuation_map, rebalance


def synthetic_param_flow_test():
    """参数透传、四因子不变与五因子下限达标。"""
    data_map, valuation_map, rebalance = _synthetic_market()
    codes = list(data_map)
    sbf.load_all_daily = lambda codes, fq="qfq": {c: data_map[c] for c in codes}
    sbf.load_valuation = lambda code: valuation_map[code]
    sbf.get_rebalance_dates = lambda codes, freq="M", fq="qfq": list(rebalance)

    calls = []
    original = sbf.build_fundamental_composite

    def spy(*args, **kwargs):
        calls.append(kwargs.get("momentum_floor", "missing"))
        return original(*args, **kwargs)

    sbf.build_fundamental_composite = spy
    common = {"codes": codes, "start_date": rebalance[0], "end_date": rebalance[-1]}
    base = sbf.run_fundamental_comparison(**common)
    floor = sbf.run_fundamental_comparison(momentum_floor=0.10, **common)
    sbf.build_fundamental_composite = original

    periods = len(base["periods"])
    assert calls == ["missing", None] * periods + ["missing", 0.10] * periods, \
        "四因子不应收到 momentum_floor，五因子应收到 0.10"
    assert base["meta"]["momentum_floor"] is None
    assert abs(floor["meta"]["momentum_floor"] - 0.10) < 1e-12
    assert "未启用动量下限" in base["meta"]["selection_note"]
    assert "动量最低权重 10%" in floor["meta"]["selection_note"]

    # 四因子与双/三因子逐值不变
    for name in ("four_factor", "dual", "three_equal", "low_pe", "low_pb"):
        assert np.allclose(base["nav"][name], floor["nav"][name], atol=0, rtol=0), \
            "%s 在启用下限后发生变化" % name
        assert np.isclose(base["metrics"][name]["annual_return"],
                          floor["metrics"][name]["annual_return"], atol=0, rtol=0)

    # 基线动量权重低于下限，启用后达到下限
    union = floor["periods"]["five_selection_rule"] == "union_top"
    assert union.any(), "合成数据应走并集分支"
    assert floor["periods"].loc[union, "five_momentum_weight"].min() >= 0.10 - 1e-12, \
        "五因子动量权重未达到 10% 下限"
    assert base["periods"].loc[union, "five_momentum_weight"].min() < 0.10 - 1e-12, \
        "基线应至少有一期低于下限，否则无法验证下限生效"
    assert not np.allclose(base["nav"]["five_factor"], floor["nav"]["five_factor"]), \
        "启用下限后五因子净值应发生变化"

    # 核心默认值不变：不传参数与显式传 None 完全一致
    trade_date = rebalance[0]
    default_call = sbf.build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True)
    explicit_none = sbf.build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True, momentum_floor=None)
    assert default_call["momentum_floor"] is None
    assert np.allclose(default_call["weights"], explicit_none["weights"], atol=0, rtol=0)
    assert default_call["selection_rule"] == explicit_none["selection_rule"]
    assert np.isclose(default_call["momentum_weight"],
                      explicit_none["momentum_weight"], atol=0, rtol=0)
    print("[PASS] 合成参数透传：四因子不变，五因子动量权重≥10%，核心默认值不变")


def synthetic_ui_control_test():
    """离屏实例化 UI，验证控件默认 10% 与仅 fundamental 传参。"""
    from PyQt5.QtWidgets import QApplication
    import ui_main

    app = QApplication.instance() or QApplication([])

    class _StubWindow:
        pass

    page = ui_main.SmartBetaPage(_StubWindow())
    assert abs(page.momentum_floor_edit.value() - 10.0) < 1e-12, "控件默认应为 10%"
    assert not page.momentum_floor_edit.isEnabled(), "非 fundamental 方案应禁用控件"
    assert page._capture_parameters()["momentum_floor"] is None

    page.strategy_combo.setCurrentIndex(3)  # fundamental
    assert page.momentum_floor_edit.isEnabled(), "fundamental 方案应启用控件"
    assert abs(page._capture_parameters()["momentum_floor"] - 0.10) < 1e-12

    page.momentum_floor_edit.setValue(15.0)
    assert abs(page._capture_parameters()["momentum_floor"] - 0.15) < 1e-12

    page.momentum_floor_edit.setValue(0.0)
    assert page._capture_parameters()["momentum_floor"] is None, "0 应等价关闭下限"

    page.strategy_combo.setCurrentIndex(1)  # three
    assert page._capture_parameters()["momentum_floor"] is None
    page.strategy_combo.setCurrentIndex(2)  # custom
    assert page._capture_parameters()["momentum_floor"] is None

    del app
    print("[PASS] UI 控件：默认 10%，仅 fundamental 生效，0 等价关闭")


if __name__ == "__main__":
    synthetic_param_flow_test()
    synthetic_ui_control_test()
    print("\n全部自测通过（仅研究用途，不构成投资建议）")
