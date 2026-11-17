# -*- coding: utf-8 -*-
"""
均线破位选股公式（纯本地，仅研究用途）。

策略口径：
- 均线组：MA5 / MA10 / MA20 / MA60，对应可配置参数 N1~N4；
- MIN_MA = 四条均线中数值最小的一条；
- 选股条件（不管四条均线如何排列，只取数值最小的那条）：
  1) 当日收盘价 < MIN_MA；
  2) 当日开盘价 < MIN_MA；
  3) 可选阴线确认（收盘价 < 开盘价），默认关闭；需要时用参数 YIN=1 开启。

公式只使用 MA/MIN/AND/OR 等通达信原生函数，不含未来函数。
公式库中的内容依赖参数 N1~N4、YIN 传入；若要粘贴到通达信使用，
可调用 tdx_text() 生成带 N1:=...; 参数声明的独立版本。

仅研究用途，不构成投资建议。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import formula_engine

DISCLAIMER = "仅研究用途，不构成投资建议"
NAME = "均线破位选股"
DEFAULT_PERIODS = {"N1": 5, "N2": 10, "N3": 20, "N4": 60}

# 公式库内容：N1~N4、YIN 由参数字典传入，保持通达信原生语法。
FORMULA = """MA1:=MA(C,N1);
MA2:=MA(C,N2);
MA3:=MA(C,N3);
MA4:=MA(C,N4);
MINMA:=MIN(MIN(MA1,MA2),MIN(MA3,MA4));
XG:C<MINMA AND O<MINMA AND (C<O OR YIN=0);"""


def default_params():
    """返回默认参数字典；YIN=0 表示不强制阴线，符合“开盘、收盘均低于最下方均线”口径。"""
    params = dict(DEFAULT_PERIODS)
    params["YIN"] = 0
    return params


def ensure_formula():
    """把公式写入本地公式库（select 类型），幂等覆盖。"""
    params = {key: {"default": value} for key, value in DEFAULT_PERIODS.items()}
    params["YIN"] = {"default": 0}
    return formula_engine.save_formula(NAME, "select", FORMULA, params=params)


def tdx_text(n1=None, n2=None, n3=None, n4=None, yin=0):
    """生成可粘贴到通达信的独立公式文本（含参数赋值，固定周期数值）。"""
    periods = [
        DEFAULT_PERIODS["N1"] if n1 is None else int(n1),
        DEFAULT_PERIODS["N2"] if n2 is None else int(n2),
        DEFAULT_PERIODS["N3"] if n3 is None else int(n3),
        DEFAULT_PERIODS["N4"] if n4 is None else int(n4),
    ]
    lines = ["{均线破位选股；%s}" % DISCLAIMER]
    lines += ["N%d:=%d;" % (index, value) for index, value in enumerate(periods, start=1)]
    lines.append("YIN:=%d;" % (1 if yin else 0))
    lines.append(FORMULA)
    return "\n".join(lines)


def extra_fields(frame, n1=None, n2=None, n3=None, n4=None):
    """计算最新交易日的 MIN_MA、收盘价、开盘价和偏离幅度（%）。

    偏离幅度 = (收盘价 / MIN_MA - 1) * 100，破位时为负值。
    数据不足以计算最长周期时返回空字典。
    """
    periods = [
        DEFAULT_PERIODS["N1"] if n1 is None else int(n1),
        DEFAULT_PERIODS["N2"] if n2 is None else int(n2),
        DEFAULT_PERIODS["N3"] if n3 is None else int(n3),
        DEFAULT_PERIODS["N4"] if n4 is None else int(n4),
    ]
    if frame is None or frame.empty or len(frame) < max(periods):
        return {}
    closes = frame["close"].astype(float)
    min_ma = min(float(closes.tail(period).mean()) for period in periods)
    last = frame.iloc[-1]
    close = float(last["close"])
    open_price = float(last["open"])
    deviation = (close / min_ma - 1) * 100 if min_ma else None
    return {
        "min_ma": min_ma,
        "close": close,
        "open": open_price,
        "deviation_pct": deviation,
    }


if __name__ == "__main__":
    # 合成行情自测：不联网，验证破位命中、阴线开关和字段计算。
    print(DISCLAIMER)
    size = 130
    close = pd.Series(np.linspace(20.0, 10.0, size))
    demo = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=size).strftime("%Y-%m-%d"),
        "open": close + 0.05,  # 阴线：开盘高于收盘
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.linspace(100000, 200000, size),
        "amount": np.linspace(1000000, 2000000, size),
    })
    params = default_params()
    result = formula_engine.FormulaEngine(demo, params).run(FORMULA, "select")
    assert bool(result.value.iloc[-1]), "下跌趋势末端应命中均线破位"

    fields = extra_fields(demo, params["N1"], params["N2"], params["N3"], params["N4"])
    assert fields["close"] < fields["min_ma"], "收盘价应低于 MIN_MA"
    assert fields["open"] < fields["min_ma"], "开盘价应低于 MIN_MA"
    assert fields["deviation_pct"] < 0, "破位偏离幅度应为负值"
    print("命中示例：", {key: round(value, 3) for key, value in fields.items()})

    # 阴线开关：最后一根改成阳线，默认 YIN=0 仍命中；手动 YIN=1 开启阴线确认后不命中。
    bullish = demo.copy()
    bullish.loc[bullish.index[-1], "open"] = float(bullish["close"].iloc[-1]) - 0.05
    assert bool(formula_engine.FormulaEngine(bullish, params).run(FORMULA, "select").value.iloc[-1])
    params_yin = dict(params, YIN=1)
    assert not bool(formula_engine.FormulaEngine(bullish, params_yin).run(FORMULA, "select").value.iloc[-1])
    print("阴线开关自测通过；通达信文本预览：")
    print(tdx_text())
    print("ma_break.py 自测通过")
