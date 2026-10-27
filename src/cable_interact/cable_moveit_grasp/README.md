# 自动抓取系统

以下四个终端均在 Mini PC 上运行。前提是相关工作区已经构建，GPU 端点云计算与发布程序已经运行，并使用相同的 ROS 通信配置。按终端 1 → 2 → 3 → 4 的顺序操作。

每个终端先退出 Conda；如果提示没有激活环境，可继续。若仍显示其他 Conda 环境名，继续执行 `conda deactivate`，直到退出。

## 终端 1：启动相机采集与发布

```bash
source /opt/ros/humble/setup.bash
source /home/flexcycle/Motion_Correlation/cmcor/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

export FASTRTPS_DEFAULT_PROFILES_FILE=/home/flexcycle/Motion_Correlation/cmcor/pointcloud/config/fastdds/mini_pc_mtu1400.xml
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"

ros2 launch cable_pointcloud mini_pc.launch.py \
  viewer:=false \
  publish_fps:=30 \
  crop_left:=100 crop_top:=140 \
  crop_width:=400 crop_height:=300
```

加载点云工作区，设置跨 PC 通信环境。相机按指定裁剪区域采集并发布 RGB-D 等信息，GPU 订阅后计算并返回 Cable 点云。`publish_fps:=30` 设置目标发布频率；`viewer:=false` 关闭独立点云查看窗口，后续统一使用抓取系统的 RViz。

保持此终端运行。

## 终端 2：启动机器人驱动并自动准备轨迹控制器

先停止旧的机器人驱动，再运行一条命令：

```bash
bash /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/start_robot.bash
```

脚本自动加载 ROS、Franka 工作区及有线 MTU1400 DDS 环境，连接真实 FR3（`192.168.3.101`），启动 `/NS_1` 下的硬件接口、控制器管理器、机器人状态/TF 发布、关节状态汇总和夹爪接口。等待状态广播器激活后，自动加载、配置并激活 `fr3_arm_controller`；若已激活则跳过。

看到 `READY: ... fr3_arm_controller active` 后再启动终端 3。控制器准备不发送轨迹或夹爪抓取目标，但会启用真实机器人控制接口；若其他 Cable 控制器占用命令接口，脚本报错，不自动停用它们。准备失败或驱动退出时脚本会清理本次启动的进程。

保持本终端运行；按 `Ctrl+C` 会一并关闭本脚本启动的驱动及其子节点。此终端不启动 MoveIt、RViz 或自动抓取动作。

## 终端 3：启动 MoveIt、抓取节点和 RViz

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash

ros2 launch cable_moveit_grasp grasp.launch.py \
  start_move_group:=true \
  joint_states_topic:=/NS_1/joint_states
```

启动 MoveIt 规划与执行接口、抓取候选生成节点、标定点云转换节点及 RViz。相机点云转换到 `fr3_link0` 后，在 RViz 中与实时机器人模型一起显示。`joint_states_topic` 指定规划使用的实时关节状态。

启动后自动将 P1～P7 的位置点拟合为有限板面障碍物 `cable_platform`，加入 MoveIt PlanningScene；RViz 的 **Obstacles** 显示该板面。总厚度默认 0.01 m，以拟合平面为中心，边界按 P1→P2→…→P7→P1 闭合。当前板面已相对原始七点沿 `fr3_link0` 的 −Y 方向平移 0.024 m（X、Z 不变）。位置点、厚度和启用状态在 `config/grasp.yaml` 的 `platform_points`、`platform_thickness`、`platform_enabled` 中配置。规划前会重新确认障碍物已加载；改变板面配置后重新生成候选。模型仅覆盖采样边界内的区域，没有自动扩展到整张实体板。

保持此终端运行。终端 2 显示 READY 后，在 RViz 选点并按终端 4 执行。

## 终端 4：选点预览并执行抓取

### 1. 加载公共环境（每个新终端一次）

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash
```

控制器已由终端 2 自动准备，夹爪默认启用，无须在此重复加载、激活或设置。公共环境只影响当前终端及之后启动的子进程。

### 2. 在 RViz 选点并预览

1. 确认点云、机器人及平台位置正确，点击 **Publish Point** 选择 Cable 抓取点。
2. 等待计算完成并显示 `Selected candidate ...; preview only`，在 **Selected Planned Trajectory** 查看抓取、+Y 抬升 40 cm 和搬运到回收点的预计轨迹。
3. 如需换候选，切换 **Interact**，右击绿色候选按钮，选择 **Select and preview this candidate**。

贴板姿态约束和目标抬高自动生效：距板面不超过 1 cm 时 TCP Z 朝基坐标 −Y；原始选点高度差小于 8 mm 时，目标沿 +Y 抬高 5 mm。具体参数在 `config/grasp.yaml` 中。

若没有成功选择候选，或规划后移动了机械臂，应重新选点规划。

### 3. 执行整套抓取回收

确认预览后执行；此命令会触发真实机械臂和夹爪动作：

```bash
ros2 service call /NS_1/cable_moveit_grasp/execute_selected std_srvs/srv/Trigger '{}'
```

动作顺序：打开夹爪 → 到达抓取点 → 抓紧 → 沿 `fr3_link0` +Y 直线抬升 40 cm → 搬运到回收姿态 → 松开。实际抬升和搬运根据测得状态重新规划，可能与预览不同；失败时停止后续流程，不发送释放动作。

驱动及抓取节点未重启时，下一轮只重复第 2、3 步。若上一轮失败且仍夹着 Cable，应先处理当前夹持状态；下一次执行会先打开夹爪。规划不包含柔性 Cable 的变形和拉扯模型。
