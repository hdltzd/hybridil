# HybridIL AUBO 独立部署程序

入口是 `deploy_aubo.py`，默认配置是 `configs/deploy_aubo.json`。程序不依赖原来的
Flexiv 控制类，也不会导入按摩机器人的业务代码，只复用了其中经过拆分的 AUBO、XJC
力传感器和导纳控制思路。

## 安全边界

默认配置为 `source.mode=replay`、`robot.allow_motion=false`，不会连接机械臂。真实运动必须
同时满足以下条件：

1. 配置改为 `source.mode=live`；
2. `pose.py` 中的 `D455_2_BASE` 和 `BASEr_2_BASE` 已经标定，
   并设置 `coordinates.calibrated=true`；
3. 配置设置 `robot.allow_motion=true`；
4. 启动命令显式增加 `--execute`；
5. AUBO 已由操作者置于允许运动状态。

程序不会自动上电、清除故障、解除保护停止或修改碰撞等级。实时模式下，XJC会按
`configs/xjc_ft.json` 的 `zero_on_startup` 配置，在开启流式读取前执行一次硬件清零；
默认值为 `true`，因此启动实时部署前必须确认传感器空载且静止。若需只读取而不清零，
将该项改为 `false`。

实时硬件进程使用轻量依赖：

```bash
conda activate aubo_massage
python -m pip install -r requirements-aubo-d455.txt
```

`pyaubo_sdk` 继续使用 AUBO 官方 SDK 在该环境中的现有安装。实时模式不再要求
`h5py` 或 `open3d`；`h5py` 只在离线 HDF5 回放时按需导入。

## 当前无硬件验证

不启动模型，直接使用 HDF5 中的真实动作检查坐标转换和控制模式：

```bash
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --recorded-actions \
  --max-policy-calls 3
```

输出中的 `motion_sent` 始终为 `false`。

## 离线模型推理

终端一先启动现有策略服务：

```bash
python node.py --config configs/node_aubo.json
```

确保 `node_aubo.json` 中的 `model_path` 指向实际 checkpoint，并且服务端口与
`deploy_aubo.json` 的 `inference.endpoint` 一致。终端二运行：

```bash
python deploy_aubo.py --config configs/deploy_aubo.json
```

这条路径读取 HDF5 点云和机器人状态，经过真实 checkpoint 推理后只打印规划结果。

## 接通硬件后的顺序

先运行 D455 的独立检查：

```bash
python tests/d455_hardware_smoke.py --frames 300 --restart-check
```

随后完成 D455 到公共 `BASE`、AUBO 基座到公共 `BASE` 的外参，以及 XJC 零偏、工具重力、质心和
`XJC_2_TCP` 标定。标定结果分别写入：

- `configs/pose.py` 的 `D455_2_BASE`（本部署代码只读取、不自动改写）；
- `configs/pose.py` 的 `BASEr_2_BASE`（在新单臂系统中，`BASEr` 就是 AUBO 基座）；
- `configs/xjc_ft.json` 的补偿参数。

矩阵命名完全沿用 `pose.py` 的规则：`A_2_B` 表示将 A 系坐标变换到 B 系。
公共 `BASE` 仍然是独立公共参考系；新单臂部署把 `BASEr` 明确定义为 AUBO 基座。运行时使用：

```python
D455_2_AUBO_BASE = np.linalg.inv(BASEr_2_BASE) @ D455_2_BASE
```

点云和模型动作仍保留在相机坐标系；只有空间裁剪时转入公共 `BASE`，执行动作时才转入
`AUBO_BASE`。`BASE_SPACE`、`OBJECT_SPACE` 和 `XJC_SPACE` 也只从 `pose.py` 读取。
离线回放来自旧 L515 数据，所以回放模式自动改用 `L515_2_BASE`，不会错误套用 D455 外参。

`pose.py` 中的 `APRILTAG_2_XJC` 用于 AprilTag 观测与 XJC 传感器之间的采集/标定链；
实时控制不通过 AprilTag 读取力，因此力补偿使用单独标定的 `XJC_2_TCP`。配置中的
`gravity_in_aubo_base` 明确表示重力向量写在 AUBO 机械臂基座坐标系，而不是公共 `BASE`。

设置 AUBO 登录密码时不写入配置文件：

```bash
export AUBO_PASSWORD='实际密码'
```

第一次联机仍保持 `allow_motion=false` 且不添加 `--execute`，只检查实时观测和模型输出。
确认坐标、力符号、工作空间和急停后，才允许空载低速执行。

当前初次联机参数有意采用保守限制：125 Hz 下 TCP 平移每周期最多 `0.1 mm`、
旋转最多 `0.001 rad`，关节目标每周期最多 `0.1°`。`control_timeout_s=0.02`
是软件看门狗：状态、力传感器、控制计算或 ServoJ 调用超时都会中止动作；位置模式也会
每周期读取并检查实测六维力。XJC 串口单次等待为 `6 ms`，以便在看门狗截止前退出。

## 控制逻辑

模型位姿在相机坐标系，先按 `pose.py` 转到公共 `BASE`，再转换到 AUBO 基坐标系后求逆运动学。模型 wrench 保持在 TCP
坐标系，与补偿后的 `external_wrench_tcp` 比较。预测力低于阈值时执行位置伺服；满足
ForceMimic 接触判定时，先将预测力投影到运动方向的正交平面，再执行导纳混合控制。
