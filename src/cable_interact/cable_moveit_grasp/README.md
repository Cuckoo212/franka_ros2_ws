# 自动抓取系统

以下四个终端均在 Mini PC 上运行。前提是相关工作区已经构建，GPU 端点云计算与发布程序已经运行，并使用相同的 ROS 通信配置。按终端 1 → 2 → 3 → 4 的顺序操作。

每个终端先退出 Conda；如果提示没有激活环境，可继续。若仍显示其他 Conda 环境名，继续执行 `conda deactivate`，直到退出。

## 终端 1：启动相机采集与发布

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/flexcycle/Motion_Correlation/cmcor/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch cable_pointcloud mini_pc.launch.py \
  viewer:=false \
  publish_fps:=30 \
  crop_left:=100 crop_top:=140 \
  crop_width:=400 crop_height:=300
```

加载点云工作区，设置跨 PC 通信环境。相机按指定裁剪区域采集并发布 RGB-D 等信息，GPU 订阅后计算并返回 Cable 点云。`publish_fps:=30` 设置目标发布频率；`viewer:=false` 关闭独立点云查看窗口，后续统一使用抓取系统的 RViz。

保持此终端运行。

## 终端 2：启动 Franka 机器人驱动

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/flexcycle/franka_ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch franka_bringup franka.launch.py \
  robot_type:=fr3 \
  robot_ip:=192.168.3.101 \
  namespace:=NS_1 \
  load_gripper:=true \
  use_fake_hardware:=false \
  controllers_yaml:=/home/flexcycle/franka_ros2_ws/install/franka_fr3_moveit_config/share/franka_fr3_moveit_config/config/fr3_ros_controllers.yaml
```

连接真实 FR3，启动硬件接口、控制器管理器、机器人状态发布和夹爪接口。所有机器人接口位于 `/NS_1` 命名空间。指定的 YAML 提供轨迹控制器参数；此启动命令本身不保证轨迹控制器已加载、激活，也不启动抓取 RViz。

保持此终端运行，等待机器人连接成功。

## 终端 3：启动 MoveIt、抓取节点和 RViz

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/flexcycle/franka_ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch cable_moveit_grasp grasp.launch.py \
  start_move_group:=true \
  joint_states_topic:=/NS_1/joint_states
```

启动 MoveIt 规划与执行接口、抓取候选生成节点、标定点云转换节点及 RViz。相机点云转换到 `fr3_link0` 后，在 RViz 中与实时机器人模型一起显示。`joint_states_topic` 指定规划使用的实时关节状态。

启动后自动将 P1～P7 的位置点拟合为有限板面障碍物 `cable_platform`，加入 MoveIt PlanningScene；RViz 的 **Obstacles** 显示该板面。总厚度默认 0.01 m，以拟合平面为中心，边界按 P1→P2→…→P7→P1 闭合。当前板面已相对原始七点沿 `fr3_link0` 的 −Y 方向平移 0.03 m（X、Z 不变）。位置点、厚度和启用状态在 `config/grasp.yaml` 的 `platform_points`、`platform_thickness`、`platform_enabled` 中配置。规划前会重新确认障碍物已加载；改变板面配置后重新生成候选。模型仅覆盖采样边界内的区域，没有自动扩展到整张实体板。

保持此终端运行。先完成终端 4 的控制器与夹爪状态检查，再在 RViz 选点。

## 终端 4：准备控制器、选点预览并执行抓取

### 1. 加载环境

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/flexcycle/franka_ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

此终端用于发送参数、检查控制器和请求执行，通信配置必须与前三个终端一致。

### 2. 检查并准备轨迹控制器

```bash
ros2 control list_controllers -c /NS_1/controller_manager
```

检查是否存在 `fr3_arm_controller`，以及其状态是否为 `active`。根据结果，只执行下面适用的一种操作。

如果列表中没有该控制器，加载、配置并激活：

```bash
ros2 run controller_manager spawner fr3_arm_controller \
  -c /NS_1/controller_manager
```

如果控制器已存在且为 `inactive`，激活：

```bash
ros2 control set_controller_state fr3_arm_controller active \
  -c /NS_1/controller_manager
```

如果已经为 `active`，直接继续，不要重复运行 spawner。若存在其他占用机械臂命令接口的 Cable 控制器，应先停用它再激活轨迹控制器。

确认最终状态和轨迹执行接口：

```bash
ros2 control list_controllers -c /NS_1/controller_manager
ros2 action info /NS_1/fr3_arm_controller/follow_joint_trajectory
```

应看到 `fr3_arm_controller` 为 `active`，对应 Action 至少有一个 server。MoveIt 将通过该接口把关节轨迹交给底层控制器。

### 3. 确认夹爪默认启用

```bash
ros2 param get /NS_1/cable_moveit_grasp gripper_enabled
ros2 action info /NS_1/franka_gripper/move
ros2 action info /NS_1/franka_gripper/grasp
```

当前代码和启动配置均默认 `gripper_enabled: true`，每次启动自动启用，不再需要手动设置。参数应返回 `Boolean value is: True`；两个夹爪 Action 均应至少有一个 server，分别用于打开夹爪和闭合抓取。

若返回 `False`，先停止终端 3 中旧的 `grasp.launch`，重新加载工作区环境并按终端 3 的命令启动，再检查参数。只重启抓取系统即可，不必重启机器人驱动；确保只运行一套抓取节点。参数通过 `ros2 param set` 设置后只对当前进程有效，重启时会重新读取启动配置。

确认 PlanningScene 中的障碍物与实际环境一致后，继续选点规划。调用执行服务前无需额外设置执行开关；选点和预览本身不会触发运动。

### 4. 在 RViz 选点、选择候选并预览

1. 确认 Cable 点云和实时机器人模型正常显示。
2. 点击顶部 **Publish Point**，再点击 Cable 上的目标抓取点。节点将使用选点时的点云生成候选抓取方向，并计算、检查轨迹。
3. 等待候选生成，切换顶部 **Interact**，在三维视图中右击候选绿色按钮，选择 **Select and preview this candidate**。
4. 在 **Selected Planned Trajectory** 显示项查看机器人动画。该动画依次显示抓取轨迹和到放置目标的预计搬运轨迹；夹爪动作由独立 Action 执行，不在该动画中演示。此时真实机械臂尚未执行轨迹。

也可在终端选择已通过检查的候选，例如编号 5：

```bash
ros2 topic pub --once \
  /NS_1/cable_moveit_grasp/select_candidate \
  std_msgs/msg/Int32 '{data: 5}'
```

将 `5` 替换为实际有效候选编号。看到 `Selected candidate ...; preview only` 表示已选择并发布预览。若没有有效候选，应重新选点；若规划后移动过机械臂，应重新生成轨迹并预览。

### 5. 执行真实抓取

确认选定轨迹合适后运行。此命令会请求真实机械臂运动及夹爪抓取：

```bash
ros2 service call \
  /NS_1/cable_moveit_grasp/execute_selected \
  std_srvs/srv/Trigger '{}'
```

节点依次打开夹爪、检查并执行抓取轨迹、闭合夹爪。抓取成功后，读取实际关节状态，用 MoveIt 重新规划并执行到 `config/grasp.yaml` 中 `place_joint_positions` 指定的七关节目标（rad），确认各关节到达误差不超过 `place_arrival_tolerance`（默认 0.10 rad，约 5.7°；规划目标容差 `place_goal_tolerance` 仍为 0.01 rad），再打开夹爪释放 Cable。

`place_enabled` 默认 `true`，因此一次请求执行整套抓取、搬运、释放。抓取后的实际搬运轨迹重新规划，可能与提前预览不同；执行前会发布该轨迹并打印阶段日志，不额外等待确认。搬运失败时不会发送释放动作；释放失败会单独报告。检查服务 `success` 和 `message` 判断整套流程结果。

规划仍只检查 PlanningScene 中已有模型，不包含柔性 Cable 变形与拉扯模型。

### 6. 再次抓取

驱动与抓取节点均未重启时，不必重复加载控制器或重复检查夹爪接口。处理好上一次夹持的 Cable，等待点云更新，再重复“选点 → 选择候选 → 检查预览 → 执行”。下一次执行前会打开夹爪，因此应先考虑当前夹持物的释放。

每次执行尝试都会清除当前选择，成功或失败后都需要重新选择候选。机器人驱动重启后重新检查控制器；抓取节点重启后夹爪默认启用，重新检查夹爪状态；实际障碍物变化后更新 PlanningScene，再重新规划。
