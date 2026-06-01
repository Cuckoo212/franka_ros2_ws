# Multi-grasp 分割运行流程

本文档记录桌面线缆 `Multi-grasp` 的完整操作顺序。当前默认配置使用：

- 工作空间：`/home/flexcycle/franka_ros2_ws`
- 代码目录：`/home/flexcycle/franka_ros2_ws/src/cable_interact`
- 机器人命名空间：`NS_1`
- 机器人 IP：`192.168.3.102`
- 控制器：`cartesian_cable_board_interact_controller`
- CMCor 数据集目录：`/home/flexcycle/Motion_Correlaion/cmcor/datasets/CMCor_multigrasp_board`
- 手眼标定文件：`/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib`

下面以 `multi_grasp/cable_000` 为例。更换线缆样本时，把命令中的 `cable_000` 改为对应目录名。

## 1. 修改代码后重新编译

仅在修改控制器或 Python 脚本后执行：

```bash
cd ~/franka_ros2_ws
colcon build --packages-select franka_example_controllers franka_bringup
source ~/franka_ros2_ws/install/setup.bash
```

## 2. 首次准备初始线缆抓取计划

这一节只需要在创建新的 `cable_XXX` 初始样本时执行一次。正式运行 Multi-grasp 时，从第 3 节开始。

### 2.1 采集 RGB-D 图像

采集前不要同时启动 ROS RealSense 节点，否则相机可能被占用。

```bash
cd ~/franka_ros2_ws/src/cable_interact
/usr/bin/python3 -m pointcloud_tools.capture_cable_rgbd
```

在图像窗口中：

- 按 `s` 保存一组图像。
- 按 `Esc` 退出。

程序会自动创建下一个目录，例如：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000
```

### 2.2 编辑线缆掩膜

采集脚本会生成一个全黑占位文件：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/mask.png
```

用sam3模型生成的mask替换

### 2.3 生成相机坐标系和机器人坐标系点云

```bash
cd ~/franka_ros2_ws/src/cable_interact
CABLE_DIR="$PWD/pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000"

/usr/bin/python3 -m pointcloud_tools.build_cable_point_cloud \
  --cable-dir "$CABLE_DIR" \
  --keep-largest-component \
  --show
```

主要输出文件：

```text
cable_camera_frame.ply
cable_robot_frame.ply
depth_vis.png
figure_camera_frame.png
figure_robot_frame.png
```

### 2.4 生成第一个抓取计划

```bash
cd ~/franka_ros2_ws/src/cable_interact
CABLE_DIR="$PWD/pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000"

/usr/bin/python3 -m pointcloud_tools.cable_on_board.compute_ply_grasp_point \
  "$CABLE_DIR/cable_robot_frame.ply" \
  --selected-pregrasp-label 3 \
  --selected-gripper-direction a \
  --show-viewer
```

生成的初始抓取计划为：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/grasp_point_cable_000
```

`--selected-pregrasp-label` 可在 `1..9` 中选择。`--selected-gripper-direction` 可选 `a` 或 `b`。

## 3. 正式运行 Multi-grasp

下面的命令需要分别在不同终端中执行，并保持前三个终端持续运行。

### 终端 1：启动机器人、控制器、RealSense 和 RViz

```bash
source ~/franka_ros2_ws/install/setup.bash
mkdir -p /tmp/ros_logs

ROS_LOG_DIR=/tmp/ros_logs ros2 launch franka_bringup example.launch.py \
  robot_ips:=192.168.3.102 \
  controller_names:=cartesian_cable_board_interact_controller
```

选择该控制器时，`example.launch.py` 默认会自动启动 RealSense 和 RViz。

### 终端 2：启动 CMCor 录制节点

```bash
source ~/franka_ros2_ws/install/setup.bash
cd ~/franka_ros2_ws/src/cable_interact

/usr/bin/python3 -m pointcloud_tools.record_cmcor_realsense_dataset --ros-args \
  -p dataset_root:=/home/flexcycle/Motion_Correlaion/cmcor/datasets/CMCor_multigrasp_board \
  -p recording_active_topic:=/NS_1/cable_interaction/recording_active \
  -p action_topic:=/NS_1/cable_interaction/action_index \
  -p ee_point_topic:=/NS_1/cable_interaction/ee_point
```

### 终端 3：启动 Multi-grasp orchestrator

```bash
source ~/franka_ros2_ws/install/setup.bash

ros2 run franka_example_controllers multigrasp_orchestrator.py --ros-args \
  -p cable_id:=cable_000 \
  -p initial_grasp_index:=1 \
  -p max_grasps:=3 \
  -p controller_node:=/NS_1/cartesian_cable_board_interact_controller \
  -p recording_active_topic:=/NS_1/cable_interaction/recording_active \
  -p dataset_root:=/home/flexcycle/Motion_Correlaion/cmcor/datasets/CMCor_multigrasp_board \
  -p current_grasp_plan_id:=multi_grasp/cable_000/grasp_point_cable_000
```

默认算法为 `correlation`。如需使用 CMCor 的 segmentation 分支，在上面的命令末尾追加：

```bash
  -p algorithm:=segmentation
```

### 终端 4：触发第一次抓取

确认前三个终端均正常运行后执行：

```bash
source ~/franka_ros2_ws/install/setup.bash

ros2 param set /NS_1/cartesian_cable_board_interact_controller \
  target_cable_id "multi_grasp/cable_000/grasp_point_cable_000"
```

第一次抓取完成后，orchestrator 会自动：

1. 读取刚刚录制的 CMCor 序列。
2. 根据运动相关性或 segmentation 结果生成线缆分割。
3. 采样新的抓取点并写入 `grasp_sample_XXX`。
4. 将新抓取计划设置到控制器，自动触发下一次抓取。
5. 达到 `max_grasps` 后停止采样新的抓取点。

例如第一次自动采样后会生成：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/grasp_sample_001/grasp_sample_001
```

## 4. 可选：在 RViz 发布初始点云

需要检查初始点云和抓取方向时，在额外终端中执行：

```bash
source ~/franka_ros2_ws/install/setup.bash
cd ~/franka_ros2_ws/src/cable_interact

/usr/bin/python3 -m pointcloud_tools.cable_on_board.publish_ply_to_rviz \
  multi_grasp/cable_000
```

## 5. 输出位置

CMCor 录制序列保存在：

```text
/home/flexcycle/Motion_Correlaion/cmcor/datasets/CMCor_multigrasp_board/motion_correlation_buffers
```

每组 Multi-grasp 序列索引保存在：

```text
/home/flexcycle/Motion_Correlaion/cmcor/datasets/CMCor_multigrasp_board/multigrasp_sequences.json
```

自动采样的抓取计划和调试掩膜保存在：

```text
~/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/grasp_sample_XXX
```

## 6. 常见问题

- `No active action frames`：检查录制节点是否使用了 `-p action_topic:=/NS_1/cable_interaction/action_index`。
- 找不到新录制序列：确认录制节点和 orchestrator 的 `dataset_root` 完全一致。
- RealSense 无法启动：确认采集脚本和 ROS RealSense 节点没有同时占用相机。
- 找不到初始抓取计划：确认 `grasp_point_cable_000` 已通过第 2.4 节生成。
- 更改脚本后运行结果未更新：重新编译 `franka_example_controllers` 并重新执行 `source ~/franka_ros2_ws/install/setup.bash`。
