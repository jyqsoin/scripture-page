# -*- coding: utf-8 -*-
"""
通达信子集公式引擎（本地、安全、无未来函数）。

支持：
- 行情变量：OPEN/O、HIGH/H、LOW/L、CLOSE/C、VOL/V/VOLUME、AMOUNT/AMO；
- 时间序列：REF/CROSS/MA/EMA/SMA/HHV/LLV/COUNT/SUM/STD/BARSLAST；
- 逻辑与运算：IF/AND/OR/NOT，以及 + - * /、> >= < <= = <>；
- 指标：MACD/KDJ/RSI/BOLL/ATR/CCI/WR/DMI/OBV/ROC/MTM/VR，另提供 KDJ_K/KDJ_D/KDJ_J、BOLL_UPPER/BOLL_LOWER、DMI_PDI/DMI_MDI/DMI_ADX；
- 中间变量 A:=...，输出变量 XG:... 或 DIF:...，语句以分号结束；
- 参数字典，例如 params={"N": 20}，公式中可写 MA(C,N)；
- 本地 JSON 公式库，并同步 SQLite formula 表。

安全约束：
- 使用 AST 白名单求值，不执行任意 Python 代码；
- 检测到 ZIG/BACKSET/PEAK/REFX/XMA 等疑似未来函数时默认拒绝运行；
- REF 负周期直接报错；所有滚动计算只使用当前和历史数据。

仅研究用途，不构成投资建议。
"""

import ast
import json
import operator
import re
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db, factors

DISCLAIMER = "仅研究用途，不构成投资建议"
FUTURE_FUNCTIONS = {
    "ZIG", "BACKSET", "PEAK", "PEAKBARS", "TROUGH", "TROUGHBARS",
    "REFX", "REFXV", "XMA", "FLATZIG", "ZIGA",
}


class FormulaError(ValueError):
    """公式语法、函数或未来函数错误。"""


@dataclass
class FormulaResult:
    """统一公式结果；value 是主输出，outputs 保存全部命名输出。"""
    kind: str
    name: str
    value: pd.Series
    outputs: dict
    warnings: list


def _strip_comments(formula):
    """去除通达信大括号注释，并兼容常见全角标点。"""
    text = re.sub(r"\{.*?\}", "", str(formula), flags=re.S)
    return (text.replace("；", ";").replace("，", ",")
            .replace("（", "(").replace("）", ")"))


def scan_future_functions(formula):
    """静态扫描疑似未来函数，返回排序后的函数名列表。"""
    text = _strip_comments(formula).upper()
    called = set(re.findall(r"([A-Z_][A-Z0-9_]*)\s*\(", text))
    return sorted(called & FUTURE_FUNCTIONS)


def _prepare_expression(expression):
    """把通达信比较/逻辑写法转换为可解析的 Python AST 表达式。"""
    text = expression.strip().replace("<>", "!=")
    text = re.sub(r"(?<![<>=!])=(?!=)", "==", text)
    # 函数形式 AND(A,B) 不带两侧空格，不会被这里替换。
    text = re.sub(r"\s+AND\s+", " and ", text, flags=re.I)
    text = re.sub(r"\s+OR\s+", " or ", text, flags=re.I)
    text = re.sub(r"\bNOT\b", "not", text, flags=re.I)
    return text


def _compile_statements(formula):
    """按分号拆句，识别 A:=中间变量、XG:输出变量和无命名主输出。"""
    text = _strip_comments(formula)
    statements = [item.strip() for item in text.split(";") if item.strip()]
    if not statements:
        raise FormulaError("公式为空")
    compiled = []
    for statement in statements:
        match = re.match(r"^([A-Za-z_]\w*)\s*(:=|:)\s*(.+)$", statement, flags=re.S)
        if match:
            target = match.group(1).upper()
            is_output = match.group(2) == ":"
            expression = match.group(3)
        else:
            target, is_output, expression = None, True, statement
        try:
            tree = ast.parse(_prepare_expression(expression), mode="eval")
        except SyntaxError as exc:
            raise FormulaError("公式语法错误：%s；原始语句：%s" % (exc, statement)) from exc
        compiled.append((target, is_output, tree))
    return compiled


class FormulaEngine:
    """对单只股票的一段日线执行公式。"""

    def __init__(self, df, params=None):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise FormulaError("行情缺少列：%s" % sorted(missing))
        self.df = df.copy()
        if "date" in self.df.columns:
            self.df = self.df.sort_values("date")
        self.df = self.df.reset_index(drop=True)
        self.index = self.df.index
        self.params = {str(k).upper(): v for k, v in (params or {}).items()}
        self.env = {
            "O": self.df["open"], "OPEN": self.df["open"],
            "H": self.df["high"], "HIGH": self.df["high"],
            "L": self.df["low"], "LOW": self.df["low"],
            "C": self.df["close"], "CLOSE": self.df["close"],
            "V": self.df["volume"], "VOL": self.df["volume"], "VOLUME": self.df["volume"],
        }
        if "amount" in self.df.columns:
            self.env["AMO"] = self.df["amount"]
            self.env["AMOUNT"] = self.df["amount"]

    def _series(self, value):
        """标量广播为与行情等长的 Series。"""
        if isinstance(value, pd.Series):
            return value if value.index.equals(self.index) else value.reindex(self.index)
        return pd.Series(value, index=self.index)

    def _bool(self, value):
        """逻辑值统一为 bool Series，NaN 视为条件不成立。"""
        return self._series(value).fillna(False).astype(bool)

    @staticmethod
    def _arity(name, args, minimum, maximum=None):
        maximum = minimum if maximum is None else maximum
        if not minimum <= len(args) <= maximum:
            expected = str(minimum) if minimum == maximum else "%d~%d" % (minimum, maximum)
            raise FormulaError("%s 需要 %s 个参数，实际 %d 个" % (name, expected, len(args)))

    @staticmethod
    def _period(value, allow_zero=False):
        """滚动周期必须是固定非负整数，不允许用未来序列动态取周期。"""
        if isinstance(value, pd.Series):
            if value.nunique(dropna=True) != 1:
                raise FormulaError("周期参数必须是固定数值")
            value = value.iloc[0]
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise FormulaError("周期参数不是整数：%s" % value) from exc
        if number < 0 or (number == 0 and not allow_zero):
            raise FormulaError("周期参数非法：%d；REF 负周期会引用未来数据" % number)
        return number

    def run(self, formula, kind="select"):
        """执行公式。kind=select 返回布尔主输出，indicator 返回数值主输出。"""
        if kind not in ("select", "indicator"):
            raise FormulaError("kind 必须是 select 或 indicator")
        future = scan_future_functions(formula)
        if future:
            raise FormulaError("检测到疑似未来函数 %s，已拒绝运行，禁止用于回测" % future)

        compiled = _compile_statements(formula)
        outputs = {}
        last_name = None
        for target, is_output, tree in compiled:
            value = self._eval(tree.body)
            if target:
                self.env[target] = value
            if is_output:
                last_name = target or "RESULT"
                outputs[last_name] = value
        if last_name is None:
            raise FormulaError("公式没有输出，请使用 XG:条件; 或指标名:表达式;")

        output_series = {name: self._series(value) for name, value in outputs.items()}
        main = output_series[last_name]
        if kind == "select":
            main = self._bool(main).rename(last_name)
        else:
            main = pd.to_numeric(main, errors="coerce").rename(last_name)
        return FormulaResult(kind, last_name, main, output_series, [])

    def _eval(self, node):
        """AST 白名单递归求值；未列出的语法一律拒绝。"""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name):
            name = node.id.upper()
            if name in self.env:
                return self.env[name]
            if name in self.params:
                return self.params[name]
            raise FormulaError("未定义变量：%s；可用参数：%s" % (name, sorted(self.params)))
        if isinstance(node, ast.BinOp):
            operations = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
            }
            function = operations.get(type(node.op))
            if function is None:
                raise FormulaError("不支持的运算符：%s" % type(node.op).__name__)
            return function(self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -self._eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return ~self._bool(self._eval(node.operand))
            raise FormulaError("不支持的一元运算符")
        if isinstance(node, ast.Compare):
            operations = {
                ast.Eq: operator.eq, ast.NotEq: operator.ne,
                ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.Lt: operator.lt, ast.LtE: operator.le,
            }
            result = None
            left = self._eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                function = operations.get(type(op))
                if function is None:
                    raise FormulaError("不支持的比较运算符")
                right = self._eval(comparator)
                current = function(left, right)
                result = current if result is None else (self._bool(result) & self._bool(current))
                left = right
            return result
        if isinstance(node, ast.BoolOp):
            values = [self._bool(self._eval(item)) for item in node.values]
            if isinstance(node.op, ast.And):
                return reduce(operator.and_, values)
            if isinstance(node.op, ast.Or):
                return reduce(operator.or_, values)
            raise FormulaError("不支持的逻辑运算符")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise FormulaError("只支持普通函数调用，不支持属性、下标或关键字参数")
            name = node.func.id.upper()
            args = [self._eval(arg) for arg in node.args]
            return self._call(name, args)
        raise FormulaError("不支持的公式语法：%s" % type(node).__name__)

    def _call(self, name, args):
        """内置函数白名单。"""
        if name == "REF":
            self._arity(name, args, 1, 2)
            return factors.ref(self._series(args[0]), self._period(args[1] if len(args) > 1 else 1))
        if name in ("MA", "EMA"):
            self._arity(name, args, 2)
            function = factors.ma if name == "MA" else factors.ema
            return function(self._series(args[0]), self._period(args[1]))
        if name == "SMA":
            self._arity(name, args, 3)
            return factors.sma_cn(self._series(args[0]), self._period(args[1]), self._period(args[2]))
        if name in ("HHV", "LLV"):
            self._arity(name, args, 2)
            function = factors.hhv if name == "HHV" else factors.llv
            return function(self._series(args[0]), self._period(args[1], allow_zero=True))
        if name == "SUM":
            self._arity(name, args, 2)
            return factors.rolling_sum(self._series(args[0]), self._period(args[1], allow_zero=True))
        if name == "STD":
            self._arity(name, args, 2)
            n = self._period(args[1])
            return self._series(args[0]).rolling(n, min_periods=n).std(ddof=0)
        if name == "COUNT":
            self._arity(name, args, 2)
            return factors.count(self._bool(args[0]), self._period(args[1], allow_zero=True))
        if name == "CROSS":
            self._arity(name, args, 2)
            return factors.cross(self._series(args[0]), self._series(args[1]))
        if name == "BARSLAST":
            self._arity(name, args, 1)
            return factors.barslast(self._bool(args[0]))
        if name == "IF":
            self._arity(name, args, 3)
            condition = self._bool(args[0])
            return pd.Series(
                np.where(condition, self._series(args[1]), self._series(args[2])), index=self.index
            )
        if name in ("AND", "OR"):
            self._arity(name, args, 2, 20)
            values = [self._bool(arg) for arg in args]
            return reduce(operator.and_ if name == "AND" else operator.or_, values)
        if name == "ABS":
            self._arity(name, args, 1)
            return self._series(args[0]).abs()
        if name in ("MAX", "MIN"):
            self._arity(name, args, 2)
            function = np.maximum if name == "MAX" else np.minimum
            return pd.Series(function(self._series(args[0]), self._series(args[1])), index=self.index)
        if name == "RSI":
            self._arity(name, args, 1, 2)
            return factors.rsi(self._series(args[0]), self._period(args[1] if len(args) > 1 else 14))
        if name == "MACD":
            self._arity(name, args, 1, 4)
            periods = [self._period(arg) for arg in args[1:]]
            fast, slow, signal = (periods + [12, 26, 9][len(periods):])
            return factors.macd(self._series(args[0]), fast, slow, signal)["macd"]
        if name in ("KDJ", "KDJ_K", "KDJ_D", "KDJ_J"):
            self._arity(name, args, 0, 3)
            periods = [self._period(arg) for arg in args]
            n, m1, m2 = (periods + [9, 3, 3][len(periods):])
            result = factors.kdj(self.df, n, m1, m2)
            return result[{"KDJ": "j", "KDJ_K": "k", "KDJ_D": "d", "KDJ_J": "j"}[name]]
        if name in ("BOLL", "BOLL_UPPER", "BOLL_LOWER"):
            self._arity(name, args, 1, 3)
            n = self._period(args[1]) if len(args) > 1 else 20
            k = float(args[2]) if len(args) > 2 else 2.0
            result = factors.boll(self._series(args[0]), n, k)
            return result[{"BOLL": "mid", "BOLL_UPPER": "upper", "BOLL_LOWER": "lower"}[name]]
        if name == "ATR":
            self._arity(name, args, 0, 1)
            return factors.atr(self.df, self._period(args[0]) if args else 14)
        if name == "CCI":
            self._arity(name, args, 0, 2)
            n = self._period(args[1]) if len(args) > 1 else 14
            return factors.cci(self.df, n)
        if name == "WR":
            self._arity(name, args, 0, 2)
            n = self._period(args[1]) if len(args) > 1 else 14
            return factors.wr(self.df, n)
        if name in ("DMI", "DMI_PDI", "DMI_MDI", "DMI_ADX"):
            self._arity(name, args, 0, 2)
            periods = [self._period(arg) for arg in args]
            n, m = (periods + [14, 6][len(periods):])
            result = factors.dmi(self.df, n, m)
            return result[{"DMI": "adx", "DMI_PDI": "pdi", "DMI_MDI": "mdi", "DMI_ADX": "adx"}[name]]
        if name == "OBV":
            self._arity(name, args, 0, 0)
            return factors.obv(self.df)
        if name == "ROC":
            self._arity(name, args, 1, 2)
            n = self._period(args[1]) if len(args) > 1 else 12
            return factors.roc(self._series(args[0]), n)
        if name == "MTM":
            self._arity(name, args, 1, 2)
            n = self._period(args[1]) if len(args) > 1 else 12
            return factors.mtm(self._series(args[0]), n)
        if name == "VR":
            self._arity(name, args, 0, 2)
            n = self._period(args[1]) if len(args) > 1 else 26
            return factors.vr(self.df, n)
        raise FormulaError("不支持的函数：%s" % name)


def select_latest(data_map, formula, params=None):
    """
    对多只股票执行选股公式，返回最后一个交易日条件成立的股票。

    data_map 格式：{code: 日线 DataFrame}。单票公式错误会被记录并继续。
    """
    selected, errors = [], {}
    for code, df in data_map.items():
        try:
            result = FormulaEngine(df, params).run(formula, "select")
            if not result.value.empty and bool(result.value.iloc[-1]):
                date = df.iloc[-1]["date"] if "date" in df.columns else ""
                selected.append({"code": code, "date": str(date)})
        except Exception as exc:
            errors[code] = str(exc)
    return selected, errors


def _formula_path(name, kind):
    """公式库路径；文件名只过滤 Windows 非法字符，保留中文。"""
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    if not safe_name:
        raise FormulaError("公式名称不能为空")
    return db.PROJECT_ROOT / "formula_lib" / kind / (safe_name + ".json")


def save_formula(name, kind, content, params=None):
    """保存公式到本地 JSON 并同步 SQLite；保存前做语法和未来函数检查。"""
    if kind not in ("select", "indicator"):
        raise FormulaError("公式类型必须是 select 或 indicator")
    future = scan_future_functions(content)
    if future:
        raise FormulaError("检测到疑似未来函数：%s" % future)
    _compile_statements(content)  # 只做静态语法验证，不依赖行情
    payload = {
        "name": name, "kind": kind, "content": content,
        "params": params or {}, "disclaimer": DISCLAIMER,
    }
    path = _formula_path(name, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    db.init_db()
    now = db.now_str()
    db.execute(
        """
        INSERT INTO formula(name, kind, content, params, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(name) DO UPDATE SET kind = excluded.kind, content = excluded.content,
            params = excluded.params, updated_at = excluded.updated_at
        """,
        (name, kind, content, json.dumps(params or {}, ensure_ascii=False), now, now),
    )
    return path


def load_formula(name, kind=None):
    """按名称读取本地公式；未指定类型时依次查选股和指标目录。"""
    kinds = [kind] if kind else ["select", "indicator"]
    for item_kind in kinds:
        path = _formula_path(name, item_kind)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("找不到本地公式：%s" % name)


def list_formulas(kind=None):
    """列出本地公式库元数据，不执行公式内容。"""
    kinds = [kind] if kind else ["select", "indicator"]
    result = []
    for item_kind in kinds:
        directory = db.PROJECT_ROOT / "formula_lib" / item_kind
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                spec = json.loads(path.read_text(encoding="utf-8"))
                result.append(spec)
            except (OSError, ValueError) as exc:
                print("公式文件读取失败 %s：%s" % (path, exc))
    return sorted(result, key=lambda item: (item.get("kind", ""), item.get("name", "")))


def run_saved_formula(name, df, params=None, kind=None):
    """读取公式库并执行；params 可覆盖公式文件中的 default 参数。"""
    spec = load_formula(name, kind)
    merged = {}
    for key, definition in (spec.get("params") or {}).items():
        merged[key.upper()] = definition.get("default") if isinstance(definition, dict) else definition
    merged.update({str(k).upper(): v for k, v in (params or {}).items()})
    return FormulaEngine(df, merged).run(spec["content"], spec["kind"])


if __name__ == "__main__":
    # 合成行情自测：不联网，覆盖选股、指标、参数化和未来函数拦截。
    print(DISCLAIMER)
    size = 120
    close = pd.Series(np.linspace(10, 20, size) + np.sin(np.arange(size) / 4))
    demo = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=size).strftime("%Y-%m-%d"),
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close,
        "volume": np.linspace(100000, 500000, size),
        "amount": np.linspace(1000000, 5000000, size),
    })

    select_formula = """
        MA5:=MA(C,5);
        MA20:=MA(C,N);
        XG:CROSS(MA5,MA20) AND C>MA20 AND VOL>MA(VOL,5);
    """
    selected = FormulaEngine(demo, {"N": 20}).run(select_formula, "select")
    print("选股主输出：", selected.name, "，最新值：", bool(selected.value.iloc[-1]))

    indicator_formula = "DIF:EMA(C,12)-EMA(C,26); DEA:EMA(DIF,9); MACD:(DIF-DEA)*2;"
    indicator = FormulaEngine(demo).run(indicator_formula, "indicator")
    assert list(indicator.outputs) == ["DIF", "DEA", "MACD"]
    print("指标输出：", list(indicator.outputs), "，MACD 最新值：", indicator.value.iloc[-1])

    builtin = FormulaEngine(demo).run("RSI14:RSI(C,14); J:KDJ_J(9,3,3); ATR14:ATR(14);", "indicator")
    assert len(builtin.outputs) == 3

    # ---- 新增指标自测 ----
    new_indicators = FormulaEngine(demo).run(
        "CCI14:CCI(14); WR14:WR(14); PDI:DMI_PDI(14,6); MDI:DMI_MDI(14,6); ADX:DMI_ADX(14,6); "
        "OBV1:OBV(); ROC12:ROC(C,12); MTM12:MTM(C,12); VR26:VR(26);",
        "indicator",
    )
    assert len(new_indicators.outputs) == 9
    for out_name in ("CCI14", "WR14", "PDI", "MDI", "ADX", "OBV1", "ROC12", "MTM12", "VR26"):
        assert out_name in new_indicators.outputs, "缺少指标输出：%s" % out_name
    print("新增指标输出：", sorted(new_indicators.outputs.keys()))
    # CCI 选股：CCI 上穿 -100
    cci_select = FormulaEngine(demo).run("XG:CROSS(CCI(14),-100);", "select")
    assert cci_select.value.dtype == bool

    try:
        FormulaEngine(demo).run("XG:ZIG(C,0.05)>REF(ZIG(C,0.05),1);", "select")
        raise AssertionError("未来函数未被拦截")
    except FormulaError as exc:
        print("未来函数拦截正常：", exc)

    try:
        FormulaEngine(demo).run("XG:C>REF(C,-1);", "select")
        raise AssertionError("REF 负周期未被拦截")
    except FormulaError as exc:
        print("负周期拦截正常：", exc)

    print("formula_engine.py 自测通过")
