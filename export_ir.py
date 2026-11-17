# -*- coding: utf-8 -*-
"""导出 Actor 到 ONNX / OpenVINO IR（FP16）。

输入契约（与训练完全一致）：
- state: (batch, 1208) float32 = 最近 60 天 × 20 因子（已做 120 日滚动 z-score、
  min_periods=60、std ddof=1、clip ±8）拼上 8 维账户状态；
- 归一化不嵌入计算图，原因见 README_NOTE：训练使用滚动统计量，且 OBV 依赖全历史
  cumsum，无法用固定均值/方差或有限窗口原样复现；部署侧需按同一口径计算
  （见 preprocess_reference.py）。
- batch 轴为动态轴，输出 action_mean 与 value，FP16 权重。

导出后做 PyTorch / ONNX Runtime / OpenVINO 三段逐值对齐，并给出 CPU 延迟。
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import torch
import yaml

from data_loader_v2 import get_feature_columns, load_stock_data
from networks_v2 import PPOAgent

ROOT = Path(__file__).resolve().parent
IR_DIR = ROOT / "ir_model"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class ExportActor(torch.nn.Module):
    """只导出确定性 Actor 均值与 Critic 价值。"""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state):
        action_mean, _, value = self.policy(state)
        return action_mean, value


def build_samples(stocks, feature_columns, lookback, count=4):
    samples = []
    meta = []
    for code in ["600519", "000001", "300750"]:
        stock = stocks[code]
        frame = stock.frame
        features = frame[feature_columns].to_numpy(dtype=np.float32)
        start = max(lookback, int(stock.split) + 5)
        for offset in range(0, count):
            index = start + offset * 17
            if index >= len(frame):
                continue
            state = features[index - lookback + 1:index + 1].reshape(-1)
            samples.append(np.concatenate([state, np.zeros(8, dtype=np.float32)]))
            meta.append({"code": code, "date": str(frame["date"].iloc[index])[:10]})
    return np.stack(samples).astype(np.float32), meta


def main():
    IR_DIR.mkdir(exist_ok=True)
    with open("config_v2.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    feature_count = int(config["network"].get("feature_count", 20))
    lookback = int(config["network"].get("lookback", 60))
    window = int(config["data"].get("normalization_window", 120))
    min_periods = int(config["data"].get("normalization_min_periods", 60))
    feature_columns = get_feature_columns(feature_count)

    checkpoint_path = ROOT / "rl_output_final_baseline" / "model_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = PPOAgent(config["network"], config["ppo"], "cpu")
    agent.policy.load_state_dict(checkpoint["model_state"])
    agent.policy.eval()
    wrapper = ExportActor(agent.policy).eval()

    stocks = load_stock_data(config["data"])
    states, meta = build_samples(stocks, feature_columns, lookback)
    state_tensor = torch.from_numpy(states)

    onnx_path = IR_DIR / "model.onnx"
    torch.onnx.export(
        wrapper,
        (state_tensor[:1],),
        str(onnx_path),
        input_names=["state"],
        output_names=["action_mean", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "action_mean": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = IR_DIR / "model.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=True)
    xml_fp32_path = IR_DIR / "model_fp32.xml"
    ov.save_model(ov_model, str(xml_fp32_path), compress_to_fp16=False)

    with torch.no_grad():
        torch_mean, torch_value = wrapper(state_tensor)
    torch_mean = torch_mean.numpy()
    torch_value = torch_value.numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_mean, ort_value = sess.run(None, {"state": states})
    ort_error = float(max(
        np.max(np.abs(ort_mean - torch_mean)), np.max(np.abs(ort_value - torch_value))
    ))

    compiled = ov.compile_model(ov_model, "CPU")
    result = compiled([states])
    ov_mean = np.asarray(result[compiled.output(0)])
    ov_value = np.asarray(result[compiled.output(1)])
    ov_error = float(max(
        np.max(np.abs(ov_mean - torch_mean)), np.max(np.abs(ov_value - torch_value))
    ))

    compiled_fp32 = ov.compile_model(ov.Core().read_model(str(xml_fp32_path)), "CPU")
    result_fp32 = compiled_fp32([states])
    fp32_mean = np.asarray(result_fp32[compiled_fp32.output(0)])
    fp32_value = np.asarray(result_fp32[compiled_fp32.output(1)])
    fp32_error = float(max(
        np.max(np.abs(fp32_mean - torch_mean)), np.max(np.abs(fp32_value - torch_value))
    ))

    batch_check = {}
    for size in [1, 4, 32]:
        batch = np.repeat(states[:1], size, axis=0)
        output = np.asarray(compiled([batch])[compiled.output(0)])
        batch_check[str(size)] = {
            "shape": list(output.shape),
            "max_abs_error_vs_single": float(np.max(np.abs(output - ov_mean[0]))),
        }

    def latency(sample, repeats=200):
        for _ in range(20):
            compiled(sample)
        start = time.perf_counter()
        for _ in range(repeats):
            compiled(sample)
        return (time.perf_counter() - start) / repeats * 1000.0

    report = {
        "version": "v3_fc_trend020",
        "source_weights": str(checkpoint_path.relative_to(ROOT)),
        "source_sha256": sha256(checkpoint_path),
        "onnx": str(onnx_path.relative_to(ROOT)),
        "xml": str(xml_path.relative_to(ROOT)),
        "bin": str((IR_DIR / "model.bin").relative_to(ROOT)),
        "xml_fp32": str(xml_fp32_path.relative_to(ROOT)),
        "bin_fp32": str((IR_DIR / "model_fp32.bin").relative_to(ROOT)),
        "precision": "FP16",
        "opset": 17,
        "inputs": {
            "state": {
                "shape": ["batch", 1208],
                "dtype": "float32",
                "layout": "60 天 × 20 因子（按日展开）[1200] + account[8]",
            }
        },
        "outputs": {
            "action_mean": {"shape": ["batch"], "dtype": "float32"},
            "value": {"shape": ["batch"], "dtype": "float32"},
        },
        "preprocessing": {
            "type": "rolling_zscore",
            "window": window,
            "min_periods": min_periods,
            "std_ddof": 1,
            "clip": [-8.0, 8.0],
            "feature_columns": feature_columns,
            "account_layout": [
                "cash_ratio", "position_fraction", "nav", "drawdown",
                "last_action", "hold_steps/60", "unrealized_pnl", "ret5",
            ],
            "note": "OBV 依赖全历史 cumsum，必须由部署侧用全历史计算，不能截断窗口",
        },
        "parity": {
            "onnx_max_abs_error": ort_error,
            "openvino_fp16_max_abs_error": ov_error,
            "openvino_fp32_max_abs_error": fp32_error,
        },
        "dynamic_batch": batch_check,
        "latency_ms": {
            "batch_1": latency(states[:1]),
            "batch_32": latency(np.repeat(states[:1], 32, axis=0), repeats=50),
        },
        "samples": [
            {
                "code": item["code"],
                "date": item["date"],
                "torch_action_mean": float(torch_mean[index]),
                "openvino_action_mean": float(ov_mean[index]),
            }
            for index, item in enumerate(meta)
        ],
        "files": {
            "onnx_bytes": onnx_path.stat().st_size,
            "xml_bytes": xml_path.stat().st_size,
            "bin_bytes": (IR_DIR / "model.bin").stat().st_size,
            "xml_fp32_bytes": xml_fp32_path.stat().st_size,
            "bin_fp32_bytes": (IR_DIR / "model_fp32.bin").stat().st_size,
        },
    }
    with open(IR_DIR / "export_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert ort_error < 1e-4, "ONNX 与 PyTorch 不一致：%s" % ort_error
    assert ov_error < 3e-3, "OpenVINO FP16 与 PyTorch 偏差过大：%s" % ov_error
    print("\nIR_EXPORT_OK")


if __name__ == "__main__":
    main()
