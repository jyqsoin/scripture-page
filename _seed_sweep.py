# -*- coding: utf-8 -*-
"""临时脚本：多随机种子跑相同配置，收集测试集 NAV 分布。"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from backtest_v2 import save_reports
from data_loader_v2 import get_feature_columns, load_stock_data
from networks_v2 import PPOAgent
from trainer_v2 import train_joint


def main():
    with open("config_v2.yaml", "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    network = base["network"]
    feature_count = int(network.get("feature_count", 20))
    get_feature_columns(feature_count)
    base["data"]["feature_count"] = feature_count
    base["env"]["lookback"] = int(network.get("lookback", 60))
    base["env"]["feature_count"] = feature_count
    stocks = load_stock_data(base["data"])

    seeds = [7, 123, 2024, 2025]
    results = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(int(base.get("torch_threads", 20)))

        output_dir = Path("rl_output_seed_%d" % seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(base, handle, allow_unicode=True, sort_keys=False)

        print("\n===== SEED %d =====" % seed, flush=True)
        agent = PPOAgent(network, base["ppo"], "cpu")
        agent = train_joint(stocks, agent, base, output_dir)
        save_reports(stocks, agent, base, output_dir)

        frame = pd.read_csv(output_dir / "backtest_report.csv")
        avg = frame[(frame["code"] == "AVERAGE") & (frame["dataset"] == "test")].iloc[0]
        item = {
            "seed": seed,
            "avg_nav": float(avg["final_nav"]),
            "avg_drawdown": float(avg["max_drawdown"]),
            "avg_position": float(avg["avg_position"]),
            "closed_total": float(frame[frame["dataset"] == "test"]["closed_trades"].sum()),
        }
        for code in ["600519", "000001", "300750"]:
            row = frame[(frame["code"] == code) & (frame["dataset"] == "test")].iloc[0]
            item["%s_nav" % code] = float(row["final_nav"])
            item["%s_dd" % code] = float(row["max_drawdown"])
        results.append(item)
        print("SEED_RESULT %s" % json.dumps(item, ensure_ascii=False), flush=True)

    with open("_seed_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print("\nSWEEP_DONE")


if __name__ == "__main__":
    main()
