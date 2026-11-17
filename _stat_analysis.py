# -*- coding: utf-8 -*-
"""临时脚本：统计检验 + 趋势参与缺口 + 基准对比。"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats

from backtest_v2 import evaluate
from data_loader_v2 import get_feature_columns, load_stock_data
from networks_v2 import PPOAgent

TARGET = 1.02


def load_agent(config):
    agent = PPOAgent(config["network"], config["ppo"], "cpu")
    checkpoint = torch.load(
        Path("rl_output_final_baseline/model_best.pth"),
        map_location="cpu",
        weights_only=False,
    )
    agent.policy.load_state_dict(checkpoint["model_state"])
    agent.policy.eval()
    return agent


def main():
    with open("config_v2.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    feature_count = int(config["network"].get("feature_count", 20))
    get_feature_columns(feature_count)
    config["data"]["feature_count"] = feature_count
    config["env"]["lookback"] = int(config["network"].get("lookback", 60))
    config["env"]["feature_count"] = feature_count

    stocks = load_stock_data(config["data"])
    agent = load_agent(config)
    lookback = config["env"]["lookback"]

    detail = {}
    for stock in stocks.values():
        env, metrics = evaluate(agent, stock, "test", config)
        view = env.view
        indexes = list(range(lookback - 1, len(view)))
        close = np.asarray(view.close, dtype=float)[indexes]
        ma20 = np.asarray(view.ma20, dtype=float)[indexes]
        momentum = np.asarray(view.momentum20, dtype=float)[indexes]
        trend = (close > ma20) & (momentum > 0)
        positions = np.asarray(env.position_history, dtype=float)[1:]
        nav = np.asarray(env.nav_history, dtype=float)
        prices = np.asarray(env.price_history, dtype=float)

        bh_nav = prices / prices[0]
        bh_peak = np.maximum.accumulate(bh_nav)
        bh_dd = float(np.max(1 - bh_nav / bh_peak))
        strat_ret = np.diff(nav) / nav[:-1]
        bh_ret = np.diff(prices) / prices[:-1]

        trend_ret_strategy = float(np.prod(1 + strat_ret[trend]) - 1)
        trend_ret_bh = float(np.prod(1 + bh_ret[trend]) - 1)
        detail[stock.code] = {
            "strategy_nav": float(metrics["final_nav"]),
            "strategy_dd": float(metrics["max_drawdown"]),
            "bh_nav": float(bh_nav[-1]),
            "bh_dd": bh_dd,
            "uptrend_share": float(trend.mean()),
            "avg_pos_uptrend": float(positions[trend].mean()) if trend.any() else 0.0,
            "avg_pos_non_uptrend": float(positions[~trend].mean()) if (~trend).any() else 0.0,
            "uptrend_return_strategy": trend_ret_strategy,
            "uptrend_return_bh": trend_ret_bh,
            "uptrend_capture": (trend_ret_strategy / trend_ret_bh) if trend_ret_bh > 0 else None,
        }

    with open("_seed_results.json", "r", encoding="utf-8") as handle:
        seeds = json.load(handle)

    navs = np.array([1.0165261013] + [item["avg_nav"] for item in seeds], dtype=float)
    n = navs.size
    mean = float(navs.mean())
    std = float(navs.std(ddof=1))
    se = std / np.sqrt(n)
    t_value, p_value = stats.ttest_1samp(navs, TARGET)
    t_crit = float(stats.t.ppf(0.975, n - 1))
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    summary = {
        "navs": navs.tolist(),
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "t": float(t_value),
        "p_two_sided": float(p_value),
        "ci95": [float(ci_low), float(ci_high)],
        "target": TARGET,
        "gap": mean - TARGET,
        "seeds": seeds,
        "detail": detail,
    }
    with open("_stat_results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("seeds", "detail")},
                     ensure_ascii=False, indent=2))
    print("DETAIL", json.dumps(detail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
