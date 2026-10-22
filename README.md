# HybridIL

<a href='https://forcemimic.github.io/'>
  <img src='https://img.shields.io/badge/Homepage-forcemimic.github.io-orange?style=flat&logo=homepage&logoColor=orange' alt='Website'>
</a>
<a href='https://arxiv.org/abs/2410.07554'>
  <img src='https://img.shields.io/badge/Arxiv-2410.07554-red?style=flat&logo=arxiv&logoColor=red' alt='Paper'>
</a>
<a href='https://2025.ieee-icra.org'>
  <img src='https://img.shields.io/badge/ICRA-2025-purple?style=flat&logo=ieee&logoColor=purple' alt='ICRA'>
</a>
<br/>

![teaser](assets/teaser.png)

[ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation](https://arxiv.org/abs/2410.07554)（ICRA 2025）中 HybridIL 的官方实现。更多信息请访问[项目主页](https://forcemimic.github.io/)。

本仓库在官方 HybridIL（Flexiv + L515 + ATI 力传感器）的基础上，扩展了**独立的 AUBO C5 + D455 + XJC 六维力传感器单臂实机部署方案**，原始训练与评估流程保持不变。

---

## 目录

- [仓库结构](#仓库结构)
- [新增 AUBO 部署代码](#新增-aubo-部署代码)
- [安装](#安装)
- [数据准备](#数据准备)
- [策略训练](#策略训练)
- [机器人评估（原 Flexiv 流程）](#机器人评估原-flexiv-流程)
- [AUBO + D455 + XJC 实机部署](#aubo--d455--xjc-实机部署)
- [测试](#测试)
- [致谢](#致谢)
- [引用](#引用)
- [许可证](#许可证)

---

## 仓库结构

```
hybridil/
├── train.py                 # 策略训练入口（robomimic diffusion_policy）
├── eval.py                  # 原 Flexiv/L515 实机评估入口（键盘控制、混合力控）
├── node.py                  # ZeroMQ 策略推理服务（已兼容 NumPy 多帧协议）
├── deploy_aubo.py           # ★ 新增：AUBO 实机部署入口（安全互锁 + 准备阶段）
├── configs/
│   ├── pose.py              # 坐标变换矩阵与裁剪空间（含 D455/XJC 新配置）
│   ├── train_dp_ftout.json  # 训练配置
│   ├── eval_dp_ftout.json   # 原 Flexiv 评估配置
│   ├── deploy_aubo.json     # ★ 新增：AUBO 部署主配置
│   ├── aubo_admit_hybrid.json # ★ 新增：软件导纳控制器参数
│   ├── xjc_ft.json          # ★ 新增：XJC 六维力传感器配置
│   └── node_aubo.json       # ★ 新增：AUBO 推理服务配置
├── controllers/
│   └── aubo_admit_hybrid.py # ★ 新增：重力补偿 + 导纳混合控制器
├── deployment/
│   ├── aubo_hybridil.py     # ★ 新增：硬件无关部署逻辑（坐标、推理客户端、动作规划）
│   └── aubo_runtime.py      # ★ 新增：单 Servo 线程、背压、遥测、统计基础设施
├── devices/
│   ├── aubo_c5.py           # ★ 新增：AUBO C5 安全 RPC 适配器
│   ├── d455.py              # ★ 新增：Intel RealSense D455 驱动
│   ├── xjc_ft.py            # ★ 新增：XJC 六维力传感器驱动（Modbus RTU / socat 桥接）
│   ├── flexiv.py            # 原 Flexiv 驱动（保留）
│   ├── l515.py              # 原 L515 相机驱动（保留）
│   └── ik.py                # 原 IK 工具（保留）
├── tools/
│   ├── aubo_telemetry_dashboard.py # ★ 新增：浏览器遥测监视 + 本机控制面板
│   └── view_object_space_web.py    # ★ 新增：D455 点云裁剪范围 Web 预览
├── tests/
│   ├── test_aubo_deployment.py     # ★ 新增：部署逻辑单元测试
│   ├── test_d455.py                # ★ 新增：D455 数学单元测试
│   ├── aubo_readonly_smoke.py      # ★ 新增：AUBO 只读连接冒烟测试
│   ├── d455_hardware_smoke.py      # ★ 新增：D455 硬件冒烟测试
│   └── xjc_hardware_smoke.py       # ★ 新增：XJC 力传感器冒烟测试
├── robomimic/              # robomimic 框架（训练/推理基础）
└── docs/AUBO_DEPLOYMENT.md # ★ 新增：AUBO 部署详细文档
```

> ★ 标记为针对 AUBO 单臂实机部署新增或大幅改动的文件；原始 Flexiv/L515 相关代码全部保留，互不影响。

---

## 新增 AUBO 部署代码

本仓库在官方实现基础上新增了一套**不依赖 Flexiv 控制类**的独立部署栈，用于将已训练的 HybridIL 模型部署到 **AUBO C5（6 自由度）＋ D455 深度相机 ＋ XJC 六维力传感器**的单臂系统：

### 设备驱动层（`devices/`）

- **`aubo_c5.py`** — AUBO C5 安全 RPC 适配器。只负责连接、登录与读取状态；**不会自动上电、解除保护停止、修改碰撞等级或发送运动命令**。提供 `inverse_kinematics`、`servo_joints` / `servo_tcp_pose`（IK + ServoJ 一步管线）、`send_command`、`move_joints`（仅准备阶段）、`stop` 等显式方法；ServoJ 队列满（code 2）作为背压单独抛出，由上层统一判断。
- **`d455.py`** — RealSense D455 最小驱动：深度对齐彩色、`depth_to_meters`、`get_point_cloud`（相机内参反投影）与纯 NumPy 体素降采样，供模型点云观测使用。
- **`xjc_ft.py`** — XJC 六维力传感器驱动：Modbus RTU 协议（从站地址 0x01/0x09），支持直连串口或自动创建 `socat` PTY 桥接到控制柜 `ser2net` 的 TCP 端口；实现硬件清零（`set_zero`）、轮询与 100/250/500 Hz 主动流式读取，带 CRC 校验与帧头同步。

### 控制器层（`controllers/aubo_admit_hybrid.py`）

- **`WrenchCompensator`** — 将 XJC 原始六维力转换为 TCP 坐标系下的力/力矩，包含手动零点偏移、基座坐标系重力向量补偿与质心力矩补偿。
- **`AuboAdmitHybridController`** — 软件导纳控制器：`加速度 = (力误差 - 阻尼×虚拟速度 + 刚度×位姿误差) / 虚拟质量`，带速度/加速度限幅、力矩误差阈值死区与 TCP Z 轴接触积分补偿。
- **`AuboDeploymentControllerManager`** — 部署运行时控制器切换：非接触 `position` 段（预求解关节轨迹）与接触 `admithybrid` 段（实时导纳修正 + 实时 IK）。

### 部署层（`deployment/`）

- **`aubo_hybridil.py`** — 硬件无关公共逻辑：
  - `CoordinateFrames`：相机 → 公共 `BASE` → AUBO 基座的坐标变换链（`D455_2_AUBO_BASE = inv(BASEr_2_BASE) @ D455_2_BASE`）；
  - `PointCloudProcessor`：按 `BASE_SPACE` / `OBJECT_SPACE` / `XJC_SPACE` 裁剪、体素降采样、固定点数采样；
  - `ActionChunkPlanner`：复刻 ForceMimic 力阈值状态机，把预测动作划分为 `position`（非接触）与 `hybrid`（接触，含正交力投影）段；
  - `InferenceClient` + NumPy/JSON 双协议 ZeroMQ 客户端；
  - `ReplaySource`：离线 HDF5 回放（使用旧 L515 数据验证坐标转换）。
- **`aubo_runtime.py`** — 实时运行基础设施：
  - `SingleServoControlLoop`：唯一 ServoJ 执行线程，100 Hz 关节路点插值 / 接触段导纳 + 实时 IK；
  - `ServoBackpressure`：ServoJ 队列满单帧丢弃、连续 5 次才停止；
  - `LatestWrenchCache`：250 Hz XJC 采集线程与样本新鲜度看门狗；
  - `TelemetryRecorder` / `RuntimeStatistics`：JSONL 遥测与线程安全统计；
  - `DashboardCommandMailbox`：接收本机浏览器仪表盘命令。

### 入口与配置

- **`deploy_aubo.py`** — AUBO 部署入口，支持多种模式：
  - `--hold-seconds N --execute`：纯位置保持测试（不连相机/模型服务）；
  - 离线回放 / `--recorded-actions`：只打印规划结果，`motion_sent=false`；
  - 实时推理不运动：默认（校验坐标与模型输出）；
  - `--execute`：真正发送 ServoJ，**必须同时满足** `allow_motion=true`、`coordinates.calibrated=true` 等安全互锁；
  - `--prepare`：低速 MoveJ 到初始化路点 → 一次策略预测 → 移动到无接触预备位姿，再进入实时控制。
- **`tools/aubo_telemetry_dashboard.py`** — 浏览器实时监视（TCP 位姿/姿态、XJC 实测力/力矩 vs 模型预测），支持软件停止、复位 init_pos、设置最大推理次数（双重开关 `--allow-control` + `browser_control.enabled`）。
- **`tools/view_object_space_web.py`** — D455 点云裁剪范围 2D/3D Web 预览，无 X11 / 外网依赖。

### 坐标约定（`configs/pose.py` 新增）

```python
D455_2_BASE      # D455 → 公共参考系（默认占位单位阵，需标定）
BASEr_2_BASE     # 单臂系统中 BASEr 即 AUBO 基座 → 公共参考系
APRILTAG_2_XJC   # AprilTag 标签 → XJC 传感器（采集/标定链，实时控制不使用）
XJC_SPACE        # XJC 力传感器工作空间裁剪边界
```

实时数据使用 `D455_2_BASE`，旧 L515 回放自动改用 `L515_2_BASE`，保证两套数据互不混淆。详细的坐标系、安全边界与验收流程见 [docs/AUBO_DEPLOYMENT.md](docs/AUBO_DEPLOYMENT.md)。

---

## 安装

```bash
git clone git@github.com:ForceMimic/hybridil.git
cd hybridil

conda create -n hil python=3.8
conda activate hil

# 策略训练
pip install -r train_requirements.txt
pip install -e .

# 原 Flexiv 实机评估
pip install -r eval_requirements.txt
# 下载 flexiv rdk v0.9.1：https://github.com/flexivrobotics/flexiv_rdk
```

### AUBO 实机部署依赖（独立环境）

```bash
conda activate aubo_massage   # 已安装 pyaubo_sdk 的环境
python -m pip install -r requirements-aubo-d455.txt
```

`pyaubo_sdk` 使用 AUBO 官方 SDK 在环境中的现有安装；`h5py` / `open3d` 仅在离线 HDF5 回放时按需导入，实时模式不再要求。

---

## 数据准备

数据采集与处理请参考 [ForceCapture](https://github.com/ForceMimic/forcecapture)。已处理数据集可从 [Google Drive](https://drive.google.com/drive/folders/1sKqOj9tO-luddM8XIaJvTFbRSEZet7C3?usp=sharing) 下载。

---

## 策略训练

```bash
python train.py --config configs/train_dp_ftout.json
```

训练使用 Diffusion Policy（`algo_name: diffusion_policy`），观测包含点云与带力标注的动作 `action_with_wrench`，checkpoint 输出到 `logs/`（如 `logs/TuiYa/.../models/model_epoch_3000.pth`）。

---

## 机器人评估（原 Flexiv 流程）

```bash
python node.py --config configs/eval_dp_ftout.json
python eval.py --config configs/eval_dp_ftout.json
```

`eval.py` 为原 Flexiv + L515 实机流程（键盘 `q` 退出、力阈值切换位置/混合控制），保持原样。

---

## AUBO + D455 + XJC 实机部署

### 1. 启动模型推理服务

```bash
python node.py --config configs/node_aubo.json
```

`configs/node_aubo.json` 中的 `model_path` 指向训练好的 checkpoint；服务端口必须与 `deploy_aubo.json` 的 `inference.endpoint`（默认 `tcp://127.0.0.1:12345`）一致。`node.py` 已支持 NumPy 多帧二进制传输点云（`inference.protocol: "numpy"`），同时兼容原 JSON 协议。

### 2. 硬件与坐标标定（首次部署必须）

1. 运行 `python tests/d455_hardware_smoke.py --frames 300` 验证相机；
2. 运行 `python tests/xjc_hardware_smoke.py --config configs/xjc_ft.json --samples 1000` 验证力传感器；
3. 标定 `pose.py` 中的 `D455_2_BASE` 与 `BASEr_2_BASE`，并设置 `coordinates.calibrated=true`；
4. 标定 `xjc_ft.json` 中的零偏、重力、质心与 `XJC_2_TCP`；
5. 设置 `robot.allow_motion=true`，并通过 `robot.password` 或环境变量 `AUBO_PASSWORD` 提供登录密码。

### 3. 按安全顺序递进执行

```bash
# 只读检查（不运动）
export AUBO_PASSWORD='实际密码'
python deploy_aubo.py --config configs/deploy_aubo.json

# 无接触准备阶段（预览 ready 位姿，move_to_policy_ready=false）
python deploy_aubo.py --config configs/deploy_aubo.json --prepare --max-policy-calls 1

# 纯位置保持测试（30 秒，不连相机/模型）
python deploy_aubo.py --config configs/deploy_aubo.json --hold-seconds 30 --execute

# 实时推理并运动（先 1 次，再 10 次，再 100 次）
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 1 --execute
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 10 --execute
```

> **安全互锁**：真实运动必须同时满足 `source.mode=live`、`coordinates.calibrated=true`、`robot.allow_motion=true`、命令行显式加 `--execute`，且操作者已在示教器确认机器人状态。程序不会自动上电、清除故障或修改碰撞等级；XJC 启动清零（`zero_on_startup=true`）前必须保证传感器空载静止。

### 4. 混合力控

将 `configs/deploy_aubo.json` 的 `execution.control_mode` 由 `position` 改为 `position_hybrid`，按 `configs/aubo_admit_hybrid.json` 调整导纳参数后执行。接触段会按预测力阈值切换到 admithybrid 软件导纳控制。

### 5. 浏览器监视（可选）

```bash
python tools/aubo_telemetry_dashboard.py --allow-control
```

需同时满足部署配置 `browser_control.enabled=true`；浏览器可实时查看预测/实测曲线，并触发软件停止、复位 init_pos 或调整最大推理次数。

> 详细的安全边界、坐标系推导、验收顺序与排障说明见 [docs/AUBO_DEPLOYMENT.md](docs/AUBO_DEPLOYMENT.md)。

---

## 测试

```bash
# 单元测试（无需硬件）
python -m unittest tests/test_aubo_deployment.py tests/test_d455.py

# 硬件冒烟测试（需对应硬件在线）
python tests/aubo_readonly_smoke.py --config configs/deploy_aubo.json   # AUBO 只读
python tests/d455_hardware_smoke.py --frames 300                        # D455
python tests/xjc_hardware_smoke.py --config configs/xjc_ft.json         # XJC
```

---

## 致谢

策略实现基于 [DexCap](https://github.com/j96w/DexCap)、[robomimic](https://github.com/ARISE-Initiative/robomimic) 与 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)，AUBO 部署驱动参考 AUBO 官方 SDK 与参考按摩机器人项目的 AUBO/XJC/导纳控制实现。感谢各位作者的杰出贡献。

---

## 引用

如果我们的工作对你有帮助，请引用：

```bibtex
@inproceedings{liu2025forcemimic,
  author={Liu, Wenhai and Wang, Junbo and Wang, Yiming and Wang, Weiming and Lu, Cewu},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  title={ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation},
  year={2025}
}
```

---

## 许可证

本仓库基于 [MIT](https://mit-license.org/) 许可证发布。
