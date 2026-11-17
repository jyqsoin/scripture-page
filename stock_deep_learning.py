# -*- coding: utf-8 -*-
"""
PyTorch A 股择时与滚动回测。

默认使用 akshare 下载后复权日线（避免长期前复权出现非正价格）：
    python stock_deep_learning.py

示例：
    python stock_deep_learning.py --code 000001 --start 20150101 --epochs 30
    python stock_deep_learning.py --objective sharpe --fee 0.0003 --slippage 0.001

无未来函数约定：
1. 当日特征只使用当日及以前的行情；未来 N 日收益只用于训练标签。
2. 每次滚动训练只使用标签已经完整实现的样本，不使用尚未到期的未来 N 日标签。
3. 当日收盘后产生信号，下一交易日开盘执行，避免按信号当日收盘价成交。
4. 特征标准化均值和标准差只从当前训练窗口估计。

训练集结果为样本内参考；测试集结果为滚动样本外结果。仅研究用途，不构成投资建议。
"""

import argparse
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
from torch.utils.data import DataLoader, Dataset

FEATURES = ["ret_1", "ret_5", "ma_diff", "vol", "vol_ratio"]
TRADING_DAYS = 242

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch A 股择时、滚动训练与含成本回测")
    parser.add_argument("--code", default="600519", help="6 位 A 股代码，默认 600519")
    parser.add_argument("--start", default="20150101", help="行情开始日期 YYYYMMDD")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"), help="行情结束日期 YYYYMMDD")
    parser.add_argument("--adjust", choices=["qfq", "hfq", "none"], default="hfq",
                        help="复权方式，默认 hfq，避免长期前复权出现非正价格")
    parser.add_argument("--horizon", type=int, default=5, help="预测未来 N 日方向，默认 5")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="末尾测试集比例，默认 0.25")
    parser.add_argument("--train-window", type=int, default=500, help="滚动训练最近样本数，默认 500")
    parser.add_argument("--retrain-every", type=int, default=60, help="每隔多少交易日重训，默认 60")
    parser.add_argument("--epochs", type=int, default=20, help="每次训练轮数，默认 20")
    parser.add_argument("--batch-size", type=int, default=64, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--threshold", type=float, default=0.55, help="上涨概率大于该值才持有，默认 0.55")
    parser.add_argument("--objective", choices=["ce", "sharpe"], default="ce",
                        help="ce=方向分类交叉熵；sharpe=可微夏普比率损失")
    parser.add_argument("--fee", type=float, default=0.0003,
                        help="单边佣金，买卖均收；万 3=0.0003，千 1=0.001")
    parser.add_argument("--slippage", type=float, default=0.001,
                        help="单边滑点，买卖均计；0.1%%=0.001")
    parser.add_argument("--stamp-duty", type=float, default=0.0005,
                        help="卖出印花税，默认 0.05%%=0.0005")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", default="stock_dl_output", help="结果输出目录")
    args = parser.parse_args()

    if len(args.code) != 6 or not args.code.isdigit():
        parser.error("--code 必须是 6 位数字股票代码")
    if args.horizon < 1 or args.train_window < 100 or args.retrain_every < 1:
        parser.error("horizon/retrain_every 必须为正数，train_window 至少为 100")
    if not 0.05 <= args.test_ratio <= 0.5:
        parser.error("--test-ratio 应在 0.05 到 0.5 之间")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs/batch_size 必须为正数，lr 必须大于 0")
    if not 0.5 <= args.threshold < 1:
        parser.error("--threshold 应在 0.5 到 1 之间")
    if min(args.fee, args.slippage, args.stamp_duty) < 0:
        parser.error("手续费、滑点和印花税不能为负数")
    if args.fee > 0.01 or args.slippage > 0.05 or args.stamp_duty > 0.01:
        parser.error("交易成本参数过大，请检查是否使用了小数而不是百分数")
    datetime.strptime(args.start, "%Y%m%d")
    datetime.strptime(args.end, "%Y%m%d")
    if args.start > args.end:
        parser.error("--start 不能晚于 --end")
    return args


def fetch_daily(code, start, end, adjust):
    """从 akshare 获取真实 A 股日线；东财失败时切换腾讯、新浪，不做等待重试。"""
    adjust_text = "" if adjust == "none" else adjust
    symbol = ("sh" if code.startswith("6") else "sz") + code

    def from_eastmoney():
        func = getattr(ak, "stock_zh_a_hist_em", None) or getattr(ak, "stock_zh_a_hist", None)
        if func is None:
            raise RuntimeError("当前 akshare 缺少东方财富日线接口")
        return func(
            symbol=code, period="daily", start_date=start, end_date=end, adjust=adjust_text,
        )

    def from_tencent():
        return ak.stock_zh_a_hist_tx(
            symbol=symbol, start_date=start, end_date=end, adjust=adjust_text,
        )

    def from_sina():
        return ak.stock_zh_a_daily_sina(
            symbol=symbol, start_date=start, end_date=end, adjust=adjust_text,
        )

    errors = []
    for source, loader in [
        ("东方财富", from_eastmoney), ("腾讯", from_tencent), ("新浪", from_sina),
    ]:
        try:
            raw = loader()
            if raw is None or raw.empty:
                raise RuntimeError("返回空数据")
            rename = {
                "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                "收盘": "close", "成交量": "volume", "成交额": "amount",
            }
            data = raw.rename(columns=rename).copy()
            required = ["date", "open", "high", "low", "close"]
            missing = [column for column in required if column not in data.columns]
            if missing:
                raise RuntimeError("行情字段缺少：%s" % missing)
            for column in ["volume", "amount"]:
                if column not in data.columns:
                    data[column] = np.nan
            data = data[required + ["volume", "amount"]]
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            for column in ["open", "high", "low", "close", "volume", "amount"]:
                data[column] = pd.to_numeric(data[column], errors="coerce")
            data = data.dropna(subset=required)
            data = data.drop_duplicates("date", keep="last").sort_values("date")
            if data.empty:
                raise RuntimeError("清洗后无有效行情")
            if (data[["open", "high", "low", "close"]] <= 0).any().any():
                raise RuntimeError("行情存在非正价格，请使用 --adjust hfq")
            print("实际行情来源：akshare/%s" % source)
            return data.reset_index(drop=True)
        except Exception as exc:
            errors.append("%s：%s" % (source, exc))
    raise RuntimeError("所有 akshare 行情接口均失败：\n" + "\n".join(errors))


def make_features(raw, horizon):
    """特征仅向后看；标签及其结束日期单独保留供训练时做隔离。"""
    data = raw.copy()
    close = data["close"]
    data["ret_1"] = close.pct_change(1)
    data["ret_5"] = close.pct_change(5)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    data["ma_diff"] = (ma5 - ma20) / ma20.replace(0, np.nan)
    data["vol"] = data["ret_1"].rolling(10).std()
    data["vol_ratio"] = data["vol"] / data["vol"].rolling(30).mean().replace(0, np.nan)

    data["fwd"] = close.shift(-horizon) / close - 1
    data["label_end_date"] = data["date"].shift(-horizon)
    data["label"] = np.where(data["fwd"].notna(), (data["fwd"] > 0).astype(float), np.nan)
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=FEATURES).reset_index(drop=True)


class ArrayDataset(Dataset):
    def __init__(self, x, y, fwd):
        self.x = x
        self.y = y
        self.fwd = fwd

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.fwd[index]


class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_model(train_data, args, seed):
    """训练一个模型；标准化参数和类别权重均只来自当前训练集。"""
    if len(train_data) < 100:
        raise RuntimeError("有效训练样本不足 100 条，请扩大行情日期范围")
    labelled = train_data.dropna(subset=["label", "fwd"])
    if len(labelled) < 100:
        raise RuntimeError("已实现标签不足 100 条，请扩大行情日期范围")

    np.random.seed(seed)
    torch.manual_seed(seed)
    x_raw = labelled[FEATURES].to_numpy(dtype=np.float32)
    y = labelled["label"].to_numpy(dtype=np.int64)
    fwd = labelled["fwd"].to_numpy(dtype=np.float32)
    mean = x_raw.mean(axis=0)
    std = np.maximum(x_raw.std(axis=0), 1e-6)
    x = (x_raw - mean) / std

    dataset = ArrayDataset(
        torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(fwd)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.objective == "ce",
        drop_last=False,
    )
    model = Net(len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    count0 = max(int((y == 0).sum()), 1)
    count1 = max(int((y == 1).sum()), 1)
    class_weight = torch.tensor(
        [len(y) / (2.0 * count0), len(y) / (2.0 * count1)], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    last_loss = float("nan")
    for _ in range(args.epochs):
        model.train()
        previous_probability = None
        losses = []
        for xb, yb, forward_return in loader:
            optimizer.zero_grad()
            logits = model(xb)
            if args.objective == "ce":
                loss = criterion(logits, yb)
            else:
                # 用上涨概率作为 0~1 的软持仓，直接最大化批次收益的夏普比率。
                probability = torch.softmax(logits, dim=1)[:, 1]
                if previous_probability is None:
                    previous = torch.cat([torch.zeros(1), probability[:-1]])
                else:
                    previous = torch.cat([previous_probability[-1:], probability[:-1]])
                change = probability - previous
                buy = torch.relu(change)
                sell = torch.relu(-change)
                soft_return = (
                    probability * forward_return
                    - buy * (args.fee + args.slippage)
                    - sell * (args.fee + args.slippage + args.stamp_duty)
                )
                sharpe = soft_return.mean() / (soft_return.std(unbiased=False) + 1e-6)
                loss = -sharpe
                previous_probability = probability.detach()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
    return model, mean, std, last_loss


@torch.no_grad()
def predict_probability(model, mean, std, data):
    model.eval()
    x = data[FEATURES].to_numpy(dtype=np.float32)
    x = (x - mean) / std
    logits = model(torch.from_numpy(x))
    return torch.softmax(logits, dim=1)[:, 1].numpy()


def make_predictions(data, split, args):
    """训练集静态拟合；测试集每 60 日仅用当时已知标签滚动重训。"""
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    train_end = train["date"].iloc[-1]
    train_fit = train[
        train["label_end_date"].notna() & (train["label_end_date"] <= train_end)
    ]
    model, mean, std, loss = train_model(train_fit, args, args.seed)
    train_probability = predict_probability(model, mean, std, train)
    print("训练集模型：样本 %d，最后损失 %.5f" % (len(train_fit), loss))

    test_probability = np.empty(len(test), dtype=np.float64)
    for block_number, offset in enumerate(range(0, len(test), args.retrain_every)):
        current = split + offset
        block_end = min(offset + args.retrain_every, len(test))
        # 标签必须在信号日前一交易日已经实现，杜绝滚动训练偷看未来。
        known_date = data["date"].iloc[current - 1]
        history = data.iloc[:current]
        fit = history[
            history["label_end_date"].notna()
            & (history["label_end_date"] <= known_date)
        ].tail(args.train_window)
        model, mean, std, loss = train_model(fit, args, args.seed + block_number + 1)
        block = data.iloc[current:split + block_end]
        test_probability[offset:block_end] = predict_probability(model, mean, std, block)
        print(
            "滚动训练 %02d：信号日 %s，最近样本 %d，最后损失 %.5f"
            % (block_number + 1, data["date"].iloc[current].strftime("%Y-%m-%d"), len(fit), loss)
        )
    return train, test, train_probability, test_probability


def build_trades(daily, fee, slippage, stamp_duty):
    """按完整开仓到平仓区间统计胜率与盈亏比，不把每个持有日当成一笔交易。"""
    position = daily["position"].to_numpy()
    dates = daily["date"].reset_index(drop=True)
    opens = daily["open"].to_numpy(dtype=float)
    trades = []
    entry = None
    for index in range(len(daily)):
        previous = position[index - 1] if index else 0.0
        current = position[index]
        if previous == 0 and current == 1:
            entry = index
        elif previous == 1 and current == 0 and entry is not None:
            entry_open = opens[entry]
            exit_open = opens[index]
            buy_price = entry_open * (1 + slippage) * (1 + fee)
            sell_price = exit_open * (1 - slippage) * (1 - fee - stamp_duty)
            trades.append({
                "entry_date": dates.iloc[entry],
                "exit_date": dates.iloc[index],
                "entry_open": entry_open,
                "exit_open": exit_open,
                "holding_days": index - entry,
                "gross_return": exit_open / entry_open - 1,
                "net_return": sell_price / buy_price - 1,
            })
            entry = None
    return pd.DataFrame(trades)


def backtest(segment, probability, args):
    """次日开盘执行的多头/空仓回测，佣金与滑点双边收取，印花税仅卖出收取。"""
    daily = segment[["date", "open", "close"]].copy().reset_index(drop=True)
    daily["probability"] = probability
    daily["signal"] = (daily["probability"] > args.threshold).astype(int)

    # 收盘信号向后移一日，在下一交易日开盘形成实际持仓；最后一日开盘强制平仓。
    position = daily["signal"].shift(1).fillna(0).to_numpy(dtype=float)
    position[-1] = 0.0
    previous = np.r_[0.0, position[:-1]]
    change = position - previous
    cost = (
        np.maximum(change, 0) * (args.fee + args.slippage)
        + np.maximum(-change, 0) * (args.fee + args.slippage + args.stamp_duty)
    )
    open_forward_return = daily["open"].shift(-1).div(daily["open"]).sub(1).fillna(0).to_numpy()

    daily["position"] = position
    daily["gross_return"] = position * open_forward_return
    daily["cost"] = cost
    daily["strategy_return"] = daily["gross_return"] - cost
    daily["nav"] = (1 + daily["strategy_return"]).cumprod()
    daily["benchmark"] = daily["open"] / daily["open"].iloc[0]
    trades = build_trades(daily, args.fee, args.slippage, args.stamp_duty)
    return daily, trades


def calculate_metrics(daily, trades):
    returns = daily["strategy_return"].to_numpy(dtype=float)
    nav = daily["nav"].to_numpy(dtype=float)
    final_nav = float(nav[-1])
    years = max(len(returns) / TRADING_DAYS, 1 / TRADING_DAYS)
    annual_return = final_nav ** (1 / years) - 1 if final_nav > 0 else -1.0
    volatility = float(returns.std())
    sharpe = float(returns.mean() / volatility * np.sqrt(TRADING_DAYS)) if volatility > 1e-12 else 0.0
    drawdown = nav / np.maximum.accumulate(nav) - 1

    if trades.empty:
        win_rate = 0.0
        profit_loss = 0.0
    else:
        net = trades["net_return"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]
        win_rate = float((net > 0).mean())
        if len(wins) and len(losses):
            profit_loss = float(wins.mean() / abs(losses.mean()))
        elif len(wins):
            profit_loss = float("inf")
        else:
            profit_loss = 0.0
    return {
        "final_nav": final_nav,
        "annual_return": float(annual_return),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": win_rate,
        "profit_loss": profit_loss,
        "trade_count": int(len(trades)),
    }


def print_metrics(name, metrics):
    profit_loss = metrics["profit_loss"]
    profit_loss_text = "∞" if np.isinf(profit_loss) else "%.2f" % profit_loss
    print(
        "[%s] 净值 %.3f | 年化 %.2f%% | 夏普 %.2f | 最大回撤 %.2f%% | "
        "胜率 %.2f%% | 盈亏比 %s | 完整交易 %d 笔"
        % (
            name, metrics["final_nav"], metrics["annual_return"] * 100,
            metrics["sharpe"], metrics["max_drawdown"] * 100,
            metrics["win_rate"] * 100, profit_loss_text, metrics["trade_count"],
        )
    )


def save_results(train_daily, test_daily, train_trades, test_trades,
                 train_metrics, test_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for axis, daily, title in [
        (axes[0], train_daily, "训练集（样本内，仅作参考）"),
        (axes[1], test_daily, "测试集（每 60 日滚动训练的样本外结果）"),
    ]:
        axis.plot(daily["date"], daily["nav"], label="策略净值", linewidth=1.5)
        axis.plot(daily["date"], daily["benchmark"], label="买入持有", linestyle="--", alpha=0.8)
        axis.axhline(1, color="gray", linewidth=0.6)
        axis.set_title(title)
        axis.set_ylabel("净值")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[-1].set_xlabel("日期")
    figure.tight_layout()
    figure.savefig(output_dir / "nav_train_test.png", dpi=160)
    plt.close(figure)

    metric_names = {
        "final_nav": "期末净值", "annual_return": "年化收益", "sharpe": "夏普比率",
        "max_drawdown": "最大回撤", "win_rate": "胜率", "profit_loss": "盈亏比",
        "trade_count": "完整交易笔数",
    }
    rows = []
    for name, metrics in [("训练集", train_metrics), ("测试集", test_metrics)]:
        rows.append({"数据集": name, **{chinese: metrics[key] for key, chinese in metric_names.items()}})
    pd.DataFrame(rows).to_csv(output_dir / "backtest_metrics.csv", index=False, encoding="utf-8-sig")

    predictions = pd.concat([
        train_daily.assign(dataset="train"), test_daily.assign(dataset="test")
    ], ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    train_trades.to_csv(output_dir / "trades_train.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    test_trades.to_csv(output_dir / "trades_test.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def main():
    args = parse_args()
    print("仅研究用途，不构成投资建议。数据源：akshare 真实 A 股日线，股票：%s" % args.code)
    print(
        "成本：单边佣金 %.4f%%，单边滑点 %.4f%%，卖出印花税 %.4f%%；训练目标：%s"
        % (args.fee * 100, args.slippage * 100, args.stamp_duty * 100, args.objective)
    )
    raw = fetch_daily(args.code, args.start, args.end, args.adjust)
    data = make_features(raw, args.horizon)
    split = int(len(data) * (1 - args.test_ratio))
    if split < 200 or len(data) - split < args.retrain_every:
        raise RuntimeError("有效行情不足：训练段至少 200 条，测试段至少覆盖一个滚动周期")

    print(
        "原始日线 %d 条，有效特征 %d 条；训练 %d 条，测试 %d 条；预测未来 %d 日方向"
        % (len(raw), len(data), split, len(data) - split, args.horizon)
    )
    train, test, train_probability, test_probability = make_predictions(data, split, args)
    train_daily, train_trades = backtest(train, train_probability, args)
    test_daily, test_trades = backtest(test, test_probability, args)
    train_metrics = calculate_metrics(train_daily, train_trades)
    test_metrics = calculate_metrics(test_daily, test_trades)

    print_metrics("训练集", train_metrics)
    print_metrics("测试集", test_metrics)
    output_dir = Path(args.output_dir)
    save_results(
        train_daily, test_daily, train_trades, test_trades,
        train_metrics, test_metrics, output_dir,
    )
    print("结果已保存到：%s" % output_dir.resolve())
    print("净值图：%s" % (output_dir / "nav_train_test.png").resolve())


if __name__ == "__main__":
    main()
