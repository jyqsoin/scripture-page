# 每日盘后影子推理：任务计划程序配置（ShadowInfer_Daily）

> 模型：v3_fc_trend020 锁版，FP16 IR；参数冻结，不调参、不写 `UPGRADE_SUCCESS.flag`
> 任务名：**ShadowInfer_Daily**　触发：每日 **16:30**　执行：影子推理 63 日滚动窗口
> 运行目录（“起始于”）：`C:\Users\17314\Documents\Trae-CN\default\aquant`
> **当前状态（2026-11-16 已创建并验证）**：`Status=Ready`，`NextRun=16:30`，手动触发 `LastTaskResult=0`；已设交流电源启动、失败重试 1 次（10 分钟）、超时 2 小时；运行账户为当前用户普通权限（未启用“最高权限”，本任务不需要）。

---

## 1. 操作配置（最终采用方案 B）

任务计划程序只配置一条操作：启动 `run_shadow_daily.bat`；脚本内部再调用 python，等价于
`.\.venv\Scripts\python.exe shadow_infer.py --days 63 --output shadow_output --refresh`，工作目录固定为 `aquant`。

| 字段 | 填写值 |
|---|---|
| 程序或脚本 | `C:\Users\17314\Documents\Trae-CN\default\aquant\run_shadow_daily.bat` |
| 添加参数 | 留空 |
| 起始于 | `C:\Users\17314\Documents\Trae-CN\default\aquant`（可留空，脚本内已 `cd /d "%~dp0"`） |

bat 提供的保障：

- 日志 `logs\shadow_YYYYMMDD.log`（`shadow start` / `shadow end exit=` / `FAILED exit=` + 完整推理输出）；
- 推理成功后自动追加监控看板（`monitor_dashboard.py`：捕获率/streak、000001 仓位、300750 超额、FP16 误差、告警）；
- 周六/周日直接退出；节假日照跑，streak 按数据截止日 `end` 去重；
- 退出码回传，任务计划程序可正确标记成功/失败。

> 备选（不加包装器）：程序填 `C:\Users\17314\Documents\Trae-CN\default\aquant\.venv\Scripts\python.exe`，参数 `shadow_infer.py --days 63 --output shadow_output --refresh`，起始于 `C:\Users\17314\Documents\Trae-CN\default\aquant`。缺点是没有日志文件、控制台输出不可见；相对路径 `.\` 在任务计划中不可用。

---

## 2. 图形界面配置步骤（截图级）

1. **打开**：`Win + R` → 输入 `taskschd.msc` → 回车（任务计划程序）。
2. **新建**：右侧「操作」栏点击 **创建任务…**（不要点“创建基本任务”，后者没有失败重试与超时设置）。
3. **常规**页
   - 名称：`ShadowInfer_Daily`
   - 描述：`AQuant 盘后影子推理（FP16 IR，只记录不调参）`
   - 安全选项：勾选 **使用最高权限运行**
   - 配置：选择 `Windows 10` / `Windows Server 2016`（有则选）
   - 若希望无人登录也运行：选 **不管用户是否登录都要运行**（保存时会要求输入当前账户密码）
4. **触发器**页 →「新建…」
   - 开始任务：**按计划每天**
   - 设置：**16:30:00**，每天，**已启用**勾选
   - 确定
5. **操作**页 →「新建…」，按第 1 节方案 A 或 B 填写三项，确定。
6. **条件**页
   - 电源：勾选 **只有在计算机使用交流电源时才启动此任务**
   - 可取消 **如果计算机改用电池电源，则停止**（避免跑一半中断）
   - 网络：一般无需勾选
7. **设置**页
   - 勾选 **允许按需运行任务**
   - 勾选 **如果任务失败，按以下频率重新启动**：`10 分钟`，**尝试 1 次**
   - 勾选 **如果任务运行时间超过以下时间，停止任务**：`2 小时`
   - 勾选 **如果请求后任务还在运行，则强制将其停止**
8. 点击 **确定** 保存；如提示输入密码，输入当前 Windows 账户密码。
9. **验证**：在任务列表选中 `ShadowInfer_Daily` → 右键 **运行** → 下方「上次运行结果」应为 `0x0`（成功）。

---

## 3. 命令行等价创建（可选）

纯命令行无法设置“起始于”，所以只适用于方案 B：

```powershell
schtasks /Create /TN "ShadowInfer_Daily" `
  /TR "\"C:\Users\17314\Documents\Trae-CN\default\aquant\run_shadow_daily.bat\"" `
  /SC DAILY /ST 16:30 /RL HIGHEST /F
```

管理命令：

```powershell
schtasks /Query  /TN "ShadowInfer_Daily" /V /FO LIST   # 查看状态与上次结果
schtasks /Run    /TN "ShadowInfer_Daily"               # 立即运行一次
schtasks /Delete /TN "ShadowInfer_Daily" /F            # 删除
```

---

## 4. 手动验证（exit=0 + 落盘）

```powershell
cd C:\Users\17314\Documents\Trae-CN\default\aquant
.\run_shadow_daily.bat
echo "exit=$LASTEXITCODE"                                   # 期望 exit=0
Get-Content .\logs\shadow_$(Get-Date -Format yyyyMMdd).log -Tail 15
Import-Csv .\shadow_output\monitor_history.csv |
  Where-Object { $_.code -eq '600519' } |
  Select-Object -Last 1 as_of,end,uptrend_capture,capture_streak_600519,max_nav_rel_diff
.\.venv\Scripts\python.exe monitor_dashboard.py --limit 10   # 独立查看看板
```

预期：日志末尾出现 `shadow end exit=0`；`monitor_history.csv` 中 600519 行的 `uptrend_capture≈0.2244`、`capture_streak_600519=1`、`max_nav_rel_diff≈1.17e-5`。

---

## 5. 告警规则（只记录，不自动调参）

| 条件 | 动作 |
|---|---|
| IR 一致性 `max_position_diff ≥ 1e-3` 或 `max_nav_rel_diff ≥ 1e-3` | 追加 `shadow_output\alerts.log`，待人工核查 |
| 600519 捕获率 `streak ≥ 5` 且当前 <40% | 追加告警，进入人工 floor 讨论 |
| 脚本非 0 退出 | `logs\shadow_*.log` 记录 `FAILED exit=...` |

> streak 按行情截止日 `end` 去重，周末/节假日重复运行不会重复计数。

---

## 6. 分工与边界

| 环节 | 硬件 | 频次 |
|---|---|---|
| 训练 / 回测 / 统计检验 | CPU 265K，20 线程 | 已锁版，不再触发 |
| 盘后批量选股推理 | Intel NPU 13 TOPS（FP16 IR） | 每交易日一次 |
| 归一化 / 账户状态 / 风控 | 部署侧 CPU，`preprocess_reference.py` | 随推理执行 |

- 归一化传完整历史（OBV 为 `cumsum`，截断会偏移），IR 不内嵌均值/方差；
- floor、止损、冷却期等参数全部冻结；调整需基于 `monitor_history.csv` 的持续证据人工决定；
- 当前 600519 捕获率 22.44%、streak=1，**不触发任何调参**。
