# -*- coding: utf-8 -*-
"""确定性回测、完整指标表、净值/仓位图与 Attention 热力图。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader_v2 import get_feature_columns
from trading_env_v2 import TradingEnvV2

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def performance_metrics(env):
    nav = np.asarray(env.nav_history, dtype=np.float64)
    returns = np.diff(nav) / np.maximum(nav[:-1], 1e-12)
    final_nav = float(nav[-1])
    annual = final_nav ** (242 / max(len(returns), 1)) - 1 if final_nav > 0 else -1.0
    sharpe = float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(242)) if len(returns) else 0.0
    drawdown = float((1 - nav / np.maximum.accumulate(nav)).max())
    trades = np.asarray(env.trade_returns, dtype=np.float64)
    wins, losses = trades[trades > 0], trades[trades < 0]
    win_rate = float(len(wins) / len(trades)) if len(trades) else 0.0
    if len(losses) == 0:
        profit_loss = 5.0 if len(wins) else 0.0
    elif len(wins) == 0:
        profit_loss = 0.0
    else:
        profit_loss = float(wins.mean() / abs(losses.mean()))
    return {
        "final_nav": final_nav,
        "total_return": final_nav - 1,
        "annual_return": float(annual),
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": float(annual / (drawdown + 1e-8)),
        "win_rate": win_rate,
        "profit_loss": profit_loss,
        "trade_count": int(env.trade_count),
        "closed_trades": int(len(trades)),
        "losing_streak": int(env.losing_streak),
        "avg_position": float(np.mean(env.position_history)),
        "fee_drag": float(getattr(env, "total_fee", 0.0) / max(env.initial_cash, 1e-8)),
    }


def evaluate(agent, stock, dataset, config):
    """最后 20% 仅传入 test；动作使用策略均值，不做训练或随机采样。"""
    env = TradingEnvV2(stock.view(dataset), config["env"])
    state = env.reset()
    while not env.done:
        action, _, _ = agent.select_action(state, deterministic=True)
        state, _, _, _ = env.step(action)
    return env, performance_metrics(env)


def _plot_nav(stock, envs, output_dir, agent_is_attention=False):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    for row, dataset in enumerate(["train", "validation", "test"]):
        env = envs[dataset]
        nav = np.asarray(env.nav_history)
        dates = pd.to_datetime(env.date_history)
        model_name = "Attention" if agent_is_attention else "FC(256)"
        axes[row].plot(dates, nav, label="PPO + " + model_name, lw=1.4)
        buy_hold = np.asarray(env.price_history) / env.price_history[0]
        axes[row].plot(dates, buy_hold, "--", label="买入持有", lw=1.0)
        axes[row].set_title("%s - %s" % (stock.code, dataset))
        axes[row].grid(alpha=0.2)
        axes[row].legend()
    fig.tight_layout()
    fig.savefig(output_dir / ("nav_%s.png" % stock.code), dpi=150)
    plt.close(fig)


def _plot_positions(env, output_dir):
    positions = np.asarray(env.position_history)
    dates = pd.to_datetime(env.date_history)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(dates, positions, lw=1.0)
    ax.fill_between(dates, positions, 0, alpha=0.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("%s 测试集仓位变化" % env.view.code)
    ax.set_ylabel("目标仓位")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / ("position_%s.png" % env.view.code), dpi=150)
    plt.close(fig)


def _plot_attention(agent, env, output_dir):
    weights = agent.attention_weights(env.last_state)
    if weights.get("time") is None or weights.get("feature") is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    time_image = axes[0].imshow(weights["time"], aspect="auto", cmap="viridis")
    axes[0].set_title("%s 时间 Attention（60日）" % env.view.code)
    axes[0].set_xlabel("Key 交易日")
    axes[0].set_ylabel("Query 交易日")
    fig.colorbar(time_image, ax=axes[0], fraction=0.046)

    feature_image = axes[1].imshow(weights["feature"], aspect="auto", cmap="magma")
    feature_columns = get_feature_columns(agent.policy.feature_count)
    axes[1].set_title("%d 因子自 Attention" % len(feature_columns))
    ticks = np.arange(0, len(feature_columns), 5)
    axes[1].set_xticks(ticks, [feature_columns[i] for i in ticks], rotation=90, fontsize=7)
    axes[1].set_yticks(ticks, [feature_columns[i] for i in ticks], fontsize=7)
    fig.colorbar(feature_image, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / ("attention_%s.png" % env.view.code), dpi=160)
    plt.close(fig)


def _plot_average_nav(test_envs, output_dir):
    series = []
    for code, env in test_envs.items():
        curve = pd.Series(
            np.asarray(env.nav_history), index=pd.to_datetime(env.date_history), name=code
        )
        series.append(curve.groupby(level=0).last())
    aligned = pd.concat(series, axis=1).sort_index().ffill().dropna()
    aligned["AVERAGE"] = aligned.mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for code in test_envs:
        ax.plot(aligned.index, aligned[code], lw=1.1, label=code)
    ax.plot(aligned.index, aligned["AVERAGE"], color="black", lw=2.0, label="三股平均")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_title("测试集净值曲线：三只股票与平均")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "nav_average.png", dpi=160)
    plt.close(fig)


def save_reports(stocks, agent, config, output_dir):
    output_dir = Path(output_dir)
    # 评估固定为推理模式，关闭 Dropout，保证回测与 NPU 推理结果可复现。
    agent.policy.eval()
    rows = []
    trade_rows = []
    test_summaries = {}
    test_envs = {}
    dataset_summaries = {"train": [], "validation": [], "test": []}
    for code, stock in stocks.items():
        envs = {}
        for dataset in ["train", "validation", "test"]:
            env, summary = evaluate(agent, stock, dataset, config)
            envs[dataset] = env
            dataset_summaries[dataset].append(summary)
            rows.append({"code": code, "dataset": dataset, **summary})
            for trade in env.trade_log:
                trade_rows.append({"code": code, "dataset": dataset, **trade})
            if dataset == "test":
                test_summaries[code] = summary
                test_envs[code] = env
                _plot_positions(env, output_dir)
                if agent.policy.use_attention:
                    _plot_attention(agent, env, output_dir)
        _plot_nav(stock, envs, output_dir, agent.policy.use_attention)

    metric_names = [
        "final_nav", "total_return", "annual_return", "sharpe", "max_drawdown", "calmar",
        "win_rate", "profit_loss", "trade_count", "closed_trades", "losing_streak",
        "avg_position", "fee_drag",
    ]
    for dataset, summaries in dataset_summaries.items():
        average = {key: float(np.mean([summary[key] for summary in summaries])) for key in metric_names}
        rows.append({"code": "AVERAGE", "dataset": dataset, **average})

    _plot_average_nav(test_envs, output_dir)
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "backtest_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        trade_rows,
        columns=["code", "dataset", "entry_date", "exit_date", "net_return", "exit_reason"],
    ).to_csv(output_dir / "closed_trades.csv", index=False, encoding="utf-8-sig")
    comparison = report.pivot(
        index="code", columns="dataset",
        values=["final_nav", "sharpe", "max_drawdown", "win_rate", "trade_count", "closed_trades"],
    )
    comparison.to_csv(output_dir / "metrics_comparison.csv", encoding="utf-8-sig")

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)
    print("\n训练/验证/测试完整对比：")
    print(report[[
        "code", "dataset", "final_nav", "annual_return", "sharpe", "max_drawdown",
        "win_rate", "profit_loss", "trade_count", "closed_trades", "avg_position",
    ]].to_string(index=False, float_format=lambda value: "%.4f" % value))

    average_nav = float(np.mean([summary["final_nav"] for summary in test_summaries.values()]))
    average_drawdown = float(np.mean([summary["max_drawdown"] for summary in test_summaries.values()]))
    success_config = config.get("success", {})
    risk_code = str(success_config.get("risk_code", "300750")).zfill(6)
    risk_drawdown = test_summaries.get(risk_code, {}).get("max_drawdown", np.inf)
    min_closed = int(success_config.get("min_closed_trades", 3))
    closed_counts = {code: int(summary["closed_trades"]) for code, summary in test_summaries.items()}
    checks = {
        "平均NAV≥%.2f" % float(success_config.get("min_average_nav", 1.02)):
            average_nav >= float(success_config.get("min_average_nav", 1.02)),
        "平均回撤<%.2f%%" % (float(success_config.get("max_average_drawdown", 0.20)) * 100):
            average_drawdown < float(success_config.get("max_average_drawdown", 0.20)),
        "%s回撤<%.2f%%" % (risk_code, float(success_config.get("max_risk_drawdown", 0.25)) * 100):
            risk_drawdown < float(success_config.get("max_risk_drawdown", 0.25)),
        "每股完整平仓≥%d" % min_closed: all(count >= min_closed for count in closed_counts.values()),
    }
    passed = all(checks.values())
    flag_path = output_dir / "UPGRADE_SUCCESS.flag"
    summary_text = (
        "测试集平均NAV：%.6f\n测试集平均回撤：%.4f%%\n%s回撤：%.4f%%\n完整平仓：%s\n"
        % (average_nav, average_drawdown * 100, risk_code, risk_drawdown * 100, closed_counts)
    )
    if passed:
        flag_path.write_text("升级成功\n" + summary_text, encoding="utf-8")
        print("\n升级成功：\n" + summary_text)
    else:
        if flag_path.exists():
            flag_path.unlink()
        failed = [name for name, ok in checks.items() if not ok]
        print("\n未达升级成功门槛：%s\n%s" % ("；".join(failed), summary_text))
    return report
