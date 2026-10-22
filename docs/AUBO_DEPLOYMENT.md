# HybridIL AUBO 独立部署程序

入口是 `deploy_aubo.py`，默认配置是 `configs/deploy_aubo.json`。程序不依赖原来的
Flexiv 控制类，也不会导入按摩机器人的业务代码，只复用了其中经过拆分的 AUBO、XJC
力传感器和导纳控制思路。

## 安全边界

真实运动必须同时满足以下条件（移植到新电脑时应先把 `allow_motion` 设为
`false` 完成只读检查）：

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

目标工控机实测运行参考 `force_sensor.py` 时，已经能够直接打开并读取
`/dev/ttyRobotTool`。这说明控制柜串口转发和虚拟设备创建已由现有系统完成，HybridIL
部署只需要用 PySerial 以 `115200` 波特率复用该设备：

```json
"port": "/dev/ttyRobotTool",
"transport": null
```

先确认设备存在且当前用户可以读写，再运行：

```bash
ls -l /dev/ttyRobotTool
python tests/xjc_hardware_smoke.py --config configs/xjc_ft.json --samples 1000
```

部署程序不会再次启动 `socat`、连接 SSH 或重启控制柜 `ser2net`，也不会删除现有的
`/dev/ttyRobotTool`。冒烟脚本只进行协议读取，不会置零传感器或发送机械臂运动。

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

默认优先读取部署配置的 ``robot.password``；也可以不填写该项而改用环境变量：

```bash
export AUBO_PASSWORD='实际密码'
```

若直接写入配置，在 ``configs/deploy_aubo.json`` 中填写：

```json
"password": "实际密码"
```

密码属于敏感信息，不应提交到 Git 仓库或上传到共享位置。

第一次联机仍保持 `allow_motion=false` 且不添加 `--execute`，只检查实时观测和模型输出。
确认坐标、力符号、工作空间和急停后，才允许空载低速执行。

当前部署基准已提高到100 Hz：TCP平移和旋转每周期最多分别为 `1.2 mm` 和
`0.0012 rad`，对应最大速度为 `0.12 m/s` 和 `0.12 rad/s`。这与此前60 Hz、
2 mm/周期的实机速度等效。位置Servo循环不读取
XJC；XJC只作为模型观测。偶发Python周期延迟会记录警告，不再立即中止动作。

## AUBO 接口与参考控制程序的关系

部署驱动的连接、接口获取、IK 和 `servoJoint` 参数沿用参考程序的 AUBO SDK 调用方式：
`RpcClient.connect(ip, 30004)`、登录、`getRobotInterface`、`inverseKinematics` 和
`servoJoint(q, 0, 0, 0.008, 0.1, 200)`。每个模型笛卡尔目标在
`AuboC5Robot.servo_tcp_pose` 中依次完成 IK 与 ServoJ；IK 或 ServoJ 出错时停止本进程
启用的伺服，不再继续发送动作。

配置中的 `robot.robot_name` 默认为 `null`，即选择控制器报告的第一个机器人；多机器人
控制柜可填写明确名称以避免选错接口。该项目不会复制参考程序中连接后自动解锁保护停止、
关闭碰撞检测、上电、回初始化位、重启运行时机器或覆写 TCP 的行为。这些状态改变必须在
示教器上由操作者确认，或以后通过单独、显式的维护程序执行。

## 控制逻辑

模型位姿在相机坐标系，先按 `pose.py` 转到公共 `BASE`，再转换到 AUBO 基坐标系后求逆运动学。模型 wrench 保持在 TCP
坐标系，与补偿后的 `external_wrench_tcp` 比较。预测力低于阈值时执行位置伺服；满足
ForceMimic 接触判定时，先将预测力投影到运动方向的正交平面，再执行导纳混合控制。

#
本对话的主要目标是：
将现有 HybridIL / ForceMimic 项目部署到新的单机械臂实机系统：
D455相机
＋
AUBO C5机械臂
＋
XJC六维力传感器
＋
已训练的HybridIL模型
并按安全顺序完成：
本地验证D455与XJC
→ 目标电脑验证AUBO连接
→ 配置模型推理与ZMQ通信
→ 标定D455到AUBO基座坐标
→ 实时推理但不运动
→ 空载低速位置控制
→ 后续再启用混合力控与完整削皮任务
当前核心原则：
保留原Flexiv/L515代码，不破坏原训练和部署流程；
新增独立的AUBO部署代码；
BASE是公共参考系，BASEr作为AUBO基座；
D455、AUBO、XJC驱动分阶段验证；
真实运动必须经过坐标标定、XJC清零、工作空间和力安全检查；
当前优先目标是目标电脑完成“真实硬件观测 → 模型推理 → 只打印动作”的闭环，而不是立刻执行削皮。
# 通用任务准备阶段

对于机械臂默认构型距离训练任务区域较远的情况，不要让HybridIL通过局部
ServoJ从远处追踪模型目标。`deploy_aubo.py`提供显式的`--prepare`流程：

```text
低速MoveJ经过init_joint_waypoints_rad
→ 采集一次实时观测并调用策略
→ 取第一预测位姿定位任务区域
→ 沿配置的工具前向轴反方向退让retreat_distance_m
→ 对ready_pose求IK并低速MoveJ到达
→ 丢弃本次预测
→ 重新采集观测并进入实时HybridIL控制
```

准备阶段有两道运动授权：`robot.allow_motion=true`和
`preparation.enabled=true`，命令行还必须同时提供`--prepare --execute`。
`init_joint_waypoints_rad`是按执行顺序排列的关节路点；MoveJ不提供避障，
这些路点必须先在现场通过示教器确认。配置中的参考AUBO `init_pos`仅为候选，
不能直接视为任何新任务的安全路径。

首次运行保持`move_to_policy_ready=false`：程序只移动到已确认的初始化路点，
打印策略生成的ready位姿和IK关节角后退出。人工核对结果与MoveJ路径后，再将
其设为`true`，第二次运行才会移动到ready位姿并进入实时控制。

```bash
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --prepare \
  --max-policy-calls 1 \
  --execute
```

如果任务物体位于TCP正Z方向，使用`tool_forward_axis: "+z"`；程序会沿TCP
负Z退让。如果实际工具前向为负Z，则配置为`"-z"`。

## 启动方式
首次从任意位置启动，需要经过初始化位置：
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --prepare \
  --max-policy-calls 100 \
  --execute
机械臂已经在任务区域，只需要重新计算预备位置：
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --prepare \
  --max-policy-calls 100 \
  --execute
机械臂已经在合适位置，直接实时推理：
python deploy_aubo.py \
  --config configs/deploy_aubo.json \
  --max-policy-calls 100 \
  --execute
# 导纳柔顺位置控制与验收（2026-08）

部署入口现在只有 `SingleServoControlLoop` 一个线程能够调用 AUBO
`servoJoint`。没有轨迹时，该线程保持最后一次成功发送的TCP位姿；有轨迹时，
它以100 Hz持续发送，并把30 Hz动作按位置线性插值和四元数SLERP生成Servo
目标。实时任务设置为 `force_position`：非接触动作段以模型TCP位姿为平衡
位姿、以零六维力为期望外力；预测力超过阈值后切换到力位混合段，使用模型
预测六维力作为期望力。两个阶段都由 `AuboAdmitHybridController` 计算本周期
TCP命令，以上一次成功关节解为IK种子，再经参考项目同名的
`send_command(joint_positions, ik_code)` 调用底层ServoJ。

阻塞式 `moveJoint` 不用于实时柔顺闭环，只在 `--prepare` 阶段低速移动到
`init_pos` 和 `ready_pose`。质量、刚度、阻尼以及导纳速度/加速度限制位于
`configs/aubo_admit_hybrid.json`。

`servoJoint` 返回 `code 2` 时按队列背压处理：丢弃当前Servo帧、不更新
最后成功目标，下一周期重新读取机器人状态。连续5次队列满才停止。其他非零
返回码仍立即停止。运行日志会输出Servo尝试/成功/队列满、最大连续队列满、
实际发送频率、RPC和控制周期耗时、XJC样本年龄/错误以及推理延迟。

XJC串口只能由250 Hz采集线程读取。策略观测读取最近一次有效缓存；样本年龄
只写入统计，不参与位置Servo停止判断。采集线程永久失败或始终没有有效样本时，
新的模型观测仍会失败并保持最后成功位姿。

60 Hz连续实机测试通过后，正式配置使用以下100 Hz基准：

```json
{
  "control_rate_hz": 100,
  "trajectory_duration_s": 0.333333,
  "control_mode": "force_position",
  "position_command": "send_command",
  "control_timeout_s": 0.1,
  "max_tcp_speed_m_s": 0.12,
  "max_tcp_angular_speed_rad_s": 0.12,
}
```

严格按以下顺序真机执行，并保存每次输出的完整JSON统计：

```bash
python deploy_aubo.py --config configs/deploy_aubo.json --hold-seconds 30 --execute
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 1 --execute
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 10 --execute
python deploy_aubo.py --config configs/deploy_aubo.json --max-policy-calls 100 --execute
```

保持模式不连接D455和 `node.py`，也不运行模型，只初始化AUBO、XJC并保持
启动瞬间的TCP位姿。它不能与 `--prepare`、回放或不带 `--execute` 的命令
组合。

部署使用NumPy多帧ZeroMQ协议传输点云，原Flexiv客户端的单帧JSON协议仍兼容。
每次策略预测的10个点会作为一个本地 `move_to_point` 轨迹块完整执行：唯一Servo
线程持续执行该块；第10点完成后，主线程才采集下一次观测并请求下一次推理。推理
期间Servo线程保持最后成功位姿，不进行速度外推。ZeroMQ REQ socket只由主线程使用。

`servoJoint`命令持续时间由程序自动设为 `1/control_rate_hz`，不再单独配置
`servo_duration_s`。100 Hz只提高Servo目标刷新频率，不提高模型的30 Hz动作频率，也不提高TCP
速度上限；若出现持续队列满或Servo错误，应先恢复60 Hz基准排查通信节拍。

不要恢复 `max_joint_step_rad`。仍保留AUBO绝对关节限位、TCP工作空间、TCP
速度/角速度、IK失败、ServoJ错误、队列背压、碰撞状态和人工急停保护。
