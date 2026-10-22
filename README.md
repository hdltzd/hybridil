# HybridIL：ForceMimic 训练与 AUBO C5 实机部署

<a href="https://forcemimic.github.io/">
  <img src="https://img.shields.io/badge/Homepage-forcemimic.github.io-orange?style=flat&logo=homepage&logoColor=orange" alt="Website">
</a>
<a href="https://arxiv.org/abs/2410.07554">
  <img src="https://img.shields.io/badge/Arxiv-2410.07554-red?style=flat&logo=arxiv&logoColor=red" alt="Paper">
</a>
<a href="https://2025.ieee-icra.org">
  <img src="https://img.shields.io/badge/ICRA-2025-purple?style=flat&logo=ieee&logoColor=purple" alt="ICRA">
</a>

![ForceMimic teaser](assets/teaser.png)

本仓库以论文 [ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation](https://arxiv.org/abs/2410.07554) 的官方 HybridIL 实现为基础，在保留原始训练流程和 Flexiv/L515 评估代码的同时，新增了一套独立的单机械臂实机部署链路：

- Intel RealSense D455：采集彩色点云；
- AUBO C5：执行模型预测的 TCP 轨迹；
- XJC 六维力传感器：提供模型观测和接触阶段的力反馈；
- HybridIL 策略：根据点云、机器人状态和力信息预测 13 维动作；
- 单线程 ServoJ 控制：以固定频率执行位置轨迹，并可切换到 TCP Z 轴力位混合控制。

当前项目的主要目标是完成削皮等接触丰富任务的训练、离线验证和安全实机部署。

## 系统数据流

```text
D455 彩色点云 ─┐
AUBO TCP 状态 ─┼─> 观测构造 ─> node.py 策略推理 ─> 13维动作序列
XJC 六维力 ────┘                                  │
                                                   ▼
                         相机系位姿 ─> AUBO基座系位姿
                                                   │
                         ┌─────────────────────────┴──────────────────────┐
                         ▼                                                ▼
                    position 模式                              position_hybrid 模式
               关节路点插值 + ServoJ                   非接触位置控制 / 接触TCP Z力控
                         └─────────────────────────┬──────────────────────┘
                                                   ▼
                                              AUBO C5
```

模型的单步动作格式为：

```text
[x, y, z, qx, qy, qz, qw, Fx, Fy, Fz, Tx, Ty, Tz]
```

其中位姿最初位于相机坐标系，六维力位于 TCP 工具坐标系。执行前，位姿通过公共 `BASE` 转换到 AUBO 基座坐标系；六维力保持在 TCP 坐标系，以便与 XJC 补偿后的测量值直接比较。

## 目录与代码职责

| 路径 | 作用 |
| --- | --- |
| `train.py` | HybridIL 策略训练入口；限制底层数值库和 PyTorch 线程，避免多进程 DataLoader 的 CPU 线程过订阅。 |
| `node.py` | 模型推理服务；加载 checkpoint，通过 ZeroMQ 接收观测并返回动作，兼容原 JSON 和新增 NumPy 二进制协议。 |
| `eval.py`、`node.py` | 保留的原 Flexiv/L515 评估流程。 |
| `deploy_aubo.py` | AUBO 部署主入口；负责配置校验、硬件初始化、准备阶段、观测/推理循环和安全退出。 |
| `deployment/aubo_hybridil.py` | 坐标转换、点云处理、动作块规划、HDF5 回放和推理通信协议。 |
| `deployment/aubo_runtime.py` | 单一 Servo 控制线程、轨迹插值、XJC 后台采集、队列背压、运行统计、遥测和浏览器命令处理。 |
| `devices/aubo_c5.py` | AUBO C5 SDK 适配层；封装连接、状态读取、IK、MoveJ、ServoJ 和错误处理。 |
| `devices/xjc_ft.py` | XJC 六维力传感器驱动；支持串口读取、Modbus RTU 协议、流式采样和可选 socat TCP 桥接。 |
| `devices/d455.py` | RealSense D455 采集与点云生成。 |
| `controllers/aubo_admit_hybrid.py` | 力/力矩补偿、六维导纳控制及当前推荐的 TCP Z 轴力位控制器。 |
| `configs/` | 训练、推理、坐标标定、AUBO、XJC 和导纳控制参数。 |
| `tests/` | 坐标、协议、控制逻辑单测以及 D455、XJC、AUBO 只读冒烟检查。 |
| `tools/view_object_space_web.py` | 在浏览器中预览 D455 点云、调整 `OBJECT_SPACE` 并导出 PLY。 |
| `tools/aubo_telemetry_dashboard.py` | 浏览器遥测面板；显示模型计划、实际 TCP 和 XJC 力数据，可选本机软件控制。 |
| `docs/AUBO_DEPLOYMENT.md` | AUBO 上机步骤、控制细节、验收顺序和故障处理说明。 |

## 环境安装

### 训练环境

原项目使用 Python 3.8：

```bash
git clone git@github.com:ForceMimic/hybridil.git
cd hybridil

conda create -n hil python=3.8
conda activate hil
python -m pip install -r train_requirements.txt
python -m pip install -e .
```

### 原 Flexiv 评估环境

```bash
python -m pip install -r eval_requirements.txt
```

此外需要安装 [Flexiv RDK v0.9.1](https://github.com/flexivrobotics/flexiv_rdk)。

### AUBO/D455 实时环境

实时部署不需要训练环境中的 PyTorch、Open3D 或 HDF5：

```bash
conda activate aubo_massage
python -m pip install -r requirements-aubo-d455.txt
```

`pyaubo_sdk` 由 AUBO 官方 SDK 提供，需要在实时环境中单独正确安装。只有离线 HDF5 回放和部署单测需要 `h5py`。

## 数据准备与策略训练

数据采集和处理请参考 [ForceCapture](https://github.com/ForceMimic/forcecapture)。原作者提供的处理后数据可以从 [Google Drive](https://drive.google.com/drive/folders/1sKqOj9tO-luddM8XIaJvTFbRSEZet7C3?usp=sharing) 下载。

本工作区将本地数据放在 `peel_data/`，该目录已加入 `.gitignore`，不会随代码提交。训练数据路径、batch size、worker 数量和输出目录由 `configs/train_dp_ftout.json` 控制：

```bash
conda activate hil
python train.py --config configs/train_dp_ftout.json
```

训练结果默认写入 `logs/<experiment>/<timestamp>/`。部署前需要将实际 checkpoint 路径填写到 `configs/node_aubo.json` 的 `model_path`。

## 原 Flexiv 评估流程

原项目入口保持不变：

```bash
python node.py --config configs/eval_dp_ftout.json
python eval.py --config configs/eval_dp_ftout.json
```

AUBO 部署不依赖 Flexiv 控制类，两条链路可以分别维护。

## AUBO 部署配置

主要配置文件如下：

| 配置 | 关键内容 |
| --- | --- |
| `configs/deploy_aubo.json` | 数据源、推理地址、相机、机器人、准备阶段、控制模式、Servo 频率、工作空间和遥测开关。 |
| `configs/node_aubo.json` | 模型 checkpoint 和推理服务端口。 |
| `configs/pose.py` | `D455_2_BASE`、`BASEr_2_BASE`、点云裁剪范围等坐标参数。 |
| `configs/xjc_ft.json` | 串口、采样率、启动清零、零偏、重力补偿、重心和 `XJC_2_TCP`。 |
| `configs/aubo_admit_hybrid.json` | 虚拟质量、刚度、阻尼、速度限制、力安全阈值和 TCP Z 力控参数。 |

项目沿用 `A_2_B` 命名规则，表示把 A 坐标系中的量转换到 B 坐标系。新单臂系统将 `BASEr` 定义为 AUBO 基座，因此：

```python
D455_2_AUBO_BASE = np.linalg.inv(BASEr_2_BASE) @ D455_2_BASE
```

接入新设备后必须重新标定，不能直接沿用其他工位的外参、工作空间或力传感器参数。

不要把真实密码提交到配置文件。建议将 `robot.password` 留空，并使用环境变量：

```bash
export AUBO_PASSWORD='实际密码'
```

## 安全机制

> **当前工作区警告：** `configs/deploy_aubo.json` 目前配置为 `source.mode=live`、`coordinates.calibrated=true`、`robot.allow_motion=true`。因此一旦命令行加入 `--execute`，程序会尝试向真实 AUBO 发送运动命令。迁移到新工位或重新标定期间，应先将 `robot.allow_motion` 改为 `false`。

真实运动需要同时满足：

1. `source.mode` 为 `live`；
2. `configs/pose.py` 中的外参已经标定，并设置 `coordinates.calibrated=true`；
3. `robot.allow_motion=true`；
4. 启动命令显式包含 `--execute`；
5. AUBO 已由操作者在示教器或控制柜上置于允许运动状态。

程序不会自动上电、清除故障、解除保护停止、关闭碰撞检测或修改控制柜安全设置。

还需注意：

- 未带 `--execute` 时不会发送机械臂运动命令，但实时模式仍会连接相机、机器人和力传感器；
- `zero_on_startup=true` 时，XJC 会在启动实时流之前执行硬件清零。启动前必须保证传感器静止且工具没有接触物体；
- `--prepare --execute` 会按配置使用阻塞式 MoveJ 经过初始化路点，这些路点没有自动避障，必须提前在现场确认；
- ServoJ 返回 code 2 被视为临时队列背压：当前帧丢弃，连续达到配置上限才停止；其他 ServoJ 错误立即停止；
- 浏览器中的“软件停止”不等于实体急停，紧急情况必须使用控制柜或示教器急停；
- 当前 `position_hybrid` 的接触控制只修正 TCP Z 方向。启用前必须确认工具接触轴、XJC 方向和力符号。

## 推荐上机顺序

### 1. 检查 D455

```bash
python tests/d455_hardware_smoke.py --frames 300 --restart-check
```

需要检查相机序列号、深度和彩色帧、点云有效点数以及设备重启后的恢复情况。

### 2. 检查 XJC

```bash
ls -l /dev/ttyRobotTool
python tests/xjc_hardware_smoke.py --config configs/xjc_ft.json --samples 1000
```

冒烟脚本只读取协议数据，不会清零传感器，也不会驱动机械臂。

### 3. 检查 AUBO 只读连接

```bash
python tests/aubo_readonly_smoke.py --config configs/deploy_aubo.json
```

只读检查不应上电或发送运动命令。

### 4. 检查点云裁剪范围

```bash
python tools/view_object_space_web.py --host 0.0.0.0 --port 8766
```

浏览器访问终端中显示的地址，确认 `BASE_SPACE`、`OBJECT_SPACE` 和工具区域能正确保留任务点云。

### 5. 启动策略服务

```bash
conda activate hil
python node.py --config configs/node_aubo.json
```

`configs/node_aubo.json` 的端口需要与 `configs/deploy_aubo.json` 中的 `inference.endpoint` 一致。

### 6. 实时观测与推理，但不运动

```bash
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --max-policy-calls 1
```

这一步会采集真实硬件观测并调用模型，但因为没有 `--execute`，只打印规划结果。先检查坐标、四元数、工作空间、点云分布、力方向和预测轨迹。

### 7. 当前位置保持测试

```bash
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --hold-seconds 30 \
  --execute
```

保持模式不连接 D455 和模型服务，只初始化 AUBO、XJC，并由唯一 Servo 线程保持当前目标。确认 Servo 发送频率、队列满次数、XJC 样本年龄和控制周期统计。

### 8. 准备阶段与短轨迹测试

首次从远离任务区的位置启动时：

```bash
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --prepare \
  --max-policy-calls 1 \
  --execute
```

准备阶段会低速经过 `init_joint_waypoints_rad`，再根据策略的第一预测位姿生成退让后的 ready pose。应先用 `move_to_policy_ready=false` 只检查 ready pose 和 IK，再经人工确认后允许移动。

完成单步验收后，再逐步增加策略调用次数：

```bash
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 1 --execute
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 10 --execute
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 100 --execute
```

不要跳过前面的只读检查、标定和低速验收步骤。

## 离线 HDF5 回放

将 `configs/deploy_aubo.json` 中的 `source.mode` 改为 `replay`，并设置正确的 `hdf5_path`、`demo` 和 `start_index`。

直接使用数据集中的记录动作检查坐标转换：

```bash
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --recorded-actions \
  --max-policy-calls 3
```

使用模型对回放观测重新推理：

```bash
python node.py --config configs/node_aubo.json
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 3
```

回放模式禁止实机运动。旧数据来自 L515 时，程序会使用对应的 L515 外参。

## 遥测面板

先在 `configs/deploy_aubo.json` 中设置：

```json
"telemetry": {
  "enabled": true,
  "path": "logs/aubo_telemetry.jsonl",
  "sample_rate_hz": 10,
  "truncate_on_start": true
}
```

启动只读面板：

```bash
python tools/aubo_telemetry_dashboard.py
```

默认地址为 `http://127.0.0.1:8765`。面板显示预测与实测 TCP 位置、姿态、力和力矩。

如需在网页中设置最大推理次数、软件停止或复位到 `init_pos`，必须同时设置部署配置的 `browser_control.enabled=true`，并显式启动：

```bash
python tools/aubo_telemetry_dashboard.py --allow-control
```

网页命令通过本机 JSONL 文件传递，不是远程机器人 SDK，也不替代实体安全系统。

## 测试

不连接硬件的部署逻辑测试：

```bash
python -m unittest tests.test_aubo_deployment
```

该测试覆盖坐标变换、动作规划、NumPy 通信协议、轨迹插值、Servo 队列背压、XJC 缓存和部署配置校验。测试环境需要安装 `h5py`、NumPy 和 SciPy。

硬件测试必须在对应设备已连接、配置正确且现场安全的环境中单独执行。完整上机说明参见 [docs/AUBO_DEPLOYMENT.md](docs/AUBO_DEPLOYMENT.md)。

## 已知边界

- `position` 是当前默认且优先验收的控制模式；`position_hybrid` 需要完成 XJC 外参、重力、重心、接触轴和力符号验证后再启用。
- `moveJoint` 只用于准备阶段，实时闭环仅由一个线程调用 ServoJ。
- 模型预测期间，Servo 线程保持最后一个成功目标，不会由推理线程直接控制机器人。
- XJC 串口只能由后台采集线程读取；策略和控制循环使用最近一次有效缓存。
- 浏览器遥测默认关闭，开启较高采样率会增加 AUBO 状态 RPC 开销。
- 工具、相机、机械臂安装位置或工位变化后，必须重新标定坐标和安全边界。

## Acknowledgement

策略实现基于 [DexCap](https://github.com/j96w/DexCap)、[robomimic](https://github.com/ARISE-Initiative/robomimic) 和 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)。感谢相关作者的工作。

## Citation

如果本项目对你的研究有帮助，请引用：

```bibtex
@inproceedings{liu2025forcemimic,
  author={Liu, Wenhai and Wang, Junbo and Wang, Yiming and Wang, Weiming and Lu, Cewu},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  title={ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation},
  year={2025}
}
```

## License

本仓库基于 [MIT License](https://mit-license.org/) 发布。
