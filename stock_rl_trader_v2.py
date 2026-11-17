# -*- coding: utf-8 -*-
"""多股联合 PPO：默认 20 因子 FC(256)，可选 Attention，一键训练入口。"""

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

from backtest_v2 import save_reports
from data_loader_v2 import get_feature_columns, load_stock_data
from networks_v2 import PPOAgent
from trainer_v2 import train_joint


def main():
    parser = argparse.ArgumentParser(description="多股联合 PPO：20因子 FC(256)，可选 Attention")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_v2.yaml")))
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 20)))
    requested_device = str(config.get("device", "cpu"))
    device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"

    network = config["network"]
    feature_count = int(network.get("feature_count", 20))
    feature_columns = get_feature_columns(feature_count)
    state_dim = (
        int(network.get("lookback", 60)) * len(feature_columns)
        + int(network.get("account_count", 8))
    )
    config["data"]["feature_count"] = feature_count
    config["env"]["lookback"] = int(network.get("lookback", 60))
    config["env"]["feature_count"] = feature_count

    output_dir = Path(config.get("output", {}).get("output_dir", "rl_output_v2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    print("=" * 76)
    model_name = "时间/因子 Attention" if network.get("use_attention", False) else "FC(256)"
    print("多股联合 PPO + %d因子 + %s，仅研究用途，不构成投资建议" % (feature_count, model_name))
    print("状态维度：%d，CPU线程：%d，设备：%s" % (state_dim, torch.get_num_threads(), device))
    print("=" * 76)
    stocks = load_stock_data(config["data"])
    for code, stock in stocks.items():
        print(
            "  %s 有效样本 %d：训练 %d / 验证 %d / 测试 %d"
            % (
                code, len(stock.frame), stock.validation_split,
                stock.split - stock.validation_split, len(stock.frame) - stock.split,
            ), flush=True,
        )

    agent = PPOAgent(network, config["ppo"], device)
    agent = train_joint(stocks, agent, config, output_dir)
    save_reports(stocks, agent, config, output_dir)
    print("\n完成。模型、训练日志、对比表和图片保存在：%s" % output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n运行失败：%s" % exc, file=sys.stderr)
        raise
