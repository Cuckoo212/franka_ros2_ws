# Multi-grasp 分割运行流程

本文档记录桌面线缆 `Multi-grasp` 的完整操作顺序。当前默认配置使用：

- 工作空间：`/home/flexcycle/franka_ros2_ws`
- 代码目录：`/home/flexcycle/franka_ros2_ws/src/cable_interact`
- 机器人命名空间：`NS_1`
- 机器人 IP：`192.168.3.112`
- 控制器：`cartesian_cable_board_interact_controller`
- CMCor 数据集目录：`/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board`
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

需要将采集结果自动发送到 GPU PC 时，先按第 7 节配置 SSH key，然后追加：

```bash
  --scp-destination flexcycle@10.157.175.101:~/Desktop/cable_rgbd/
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

用 SAM3 模型生成线缆前景 mask，替换占位文件：

```bash
cd ~/franka_ros2_ws/src/cable_interact
~/miniconda3/envs/sam3/bin/python -m pointcloud_tools.generate_cable_mask_sam3 \
  --cable-id cable_000
```

脚本默认读取：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/rgb_000.png
```

并生成：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/mask.png
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/mask_overlay.png
```

创建新的样本时，只需要把 `--cable-id cable_000` 改为对应目录名，例如
`--cable-id cable_001` 会读取 `rgb_001.png`。如果 RGB 文件序号和目录序号
不同，可额外指定 `--rgb-id rgb_123`。

默认行为是合并 SAM3 对文本提示 `cable` 返回的全部候选。如果图像中有多根
线缆，但只需要最高分实例，可追加：

```bash
  --selection best
```

### 2.3 生成相机坐标系和机器人坐标系点云

```bash
cd ~/franka_ros2_ws/src/cable_interact
CABLE_DIR="$PWD/pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_002"

/usr/bin/python3 -m pointcloud_tools.build_cable_point_cloud \
  --cable-dir "$CABLE_DIR" \
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
CABLE_DIR="$PWD/pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_002"

/usr/bin/python3 -m pointcloud_tools.cable_on_board.compute_ply_grasp_point \
  "$CABLE_DIR/cable_robot_frame.ply" \
  --selected-grasp-label E1 \
  --endpoint-offset 0.20 \
  --show-viewer
```

生成的初始抓取计划为：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_002/grasp_point_cable_002
```

程序将板面点云细化为骨架，检测全部骨架端点，并从每个端点沿骨架向内生成 20 cm 抓取候选，
按端点坐标稳定编号为 `E1`、`E2`、……。查看候选后通过 `--selected-grasp-label E<N>` 选择实际目标；
编号只表示空间顺序，不表示候选属于哪一根 cable。计算阶段只保存所选 E 点对应的两个几何朝向候选
`a` 和 `b`，并将最终 gripper frame 标记为 `pending`；不需要传入 `--selected-gripper-direction`。
运行控制器后通过参数 `target_gripper_direction` 选择 `a` 或 `b`。明确选择时，控制器直接使用该
target frame，不再按 joint 7 限位余量自动选择；设为 `auto` 时才保留自动选择逻辑。

## 3. 正式运行 Multi-grasp

下面使用 Mini PC 控制机器人和采集数据，3090 机器计算 cable vote 与下一抓取点。Mini PC 上的
orchestrator 只负责提交任务、等待 3090 回传结果并触发控制器，不运行 CMCor 推理。

### Mini PC 终端 1：先启动机器人、RealSense、RViz 和点云发布器

```bash
source ~/franka_ros2_ws/install/setup.bash
mkdir -p /tmp/ros_logs

ROS_LOG_DIR=/tmp/ros_logs ros2 launch franka_bringup example.launch.py \
  robot_ips:=192.168.3.112 \
  controller_names:=cartesian_cable_board_interact_controller \
  use_rviz:=true \
  launch_realsense:=true \
  spawn_controller:=false &

/usr/bin/python3 \
  /home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/cable_on_board/publish_ply_to_rviz.py \
  multi_grasp/cable_002 &

wait
```

`controller_names` 在这里用于选择 cable 专用 RViz 配置，但由于 `spawn_controller:=false`，
此时还没有加载 cable interaction 控制器。先在 RViz 中确认机器人状态、点云和坐标系正常，
并确认机械臂周围安全，再打开终端 2。点云发布器读取：

```text
/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_002/cable_robot_frame.ply
/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_002/grasp_point_cable_002
```

它在 `/ply_cloud` 发布机器人坐标系点云，并在 `/cable_grasp_vectors` 发布：

- 所有编号为 `E1`、`E2`、……的预抓取候选点及标签；
- 当前 `--selected-grasp-label` 对应的 `grasp_point`；
- 该抓取点处橙色的 `a` 和青色的 `b` 两个候选 gripper 朝向及文字标签；
- `candidate_tcp_frame_0` 和 `candidate_tcp_frame_1` 两个候选 TCP 坐标系。
- 位于 target 局部 `-Z` 方向 15 cm 的 `candidate_hover_frame_0/1`。两个候选 hover 的位置相同，
  XY 朝向分别对应 `a/b`，Z 轴均与 target Z 轴一致。

控制器收到目标后使用 `target_gripper_direction` 指定的 `a/b`（或在 `auto` 模式下自动选择），把最终 TCP 三轴写回抓取计划，
点云发布器随后发布 `target_tcp_frame` 和实际的 `hover_target_frame`。控制器使用 link0 坐标系中的公式
`p_hover = p_target - 0.15 * z_target`，并令 hover 与 target 的朝向完全相同；因此从 hover 到 target
只沿 target Z 轴做 15 cm 直线接近，不再在接近过程中旋转夹爪。

如果重新运行 `compute_ply_grasp_point` 并修改了 `grasp_point_cable_002`，发布器会在运行期间重新读取
抓取计划，RViz 中的 E 候选点和最终抓取点会自动更新，不需要重启发布器。

### Mini PC 终端 2：启动 CMCor 录制节点

```bash
source ~/franka_ros2_ws/install/setup.bash
cd ~/franka_ros2_ws/src/cable_interact

/usr/bin/python3 -m pointcloud_tools.record_cmcor_realsense_dataset --ros-args \
  -p dataset_root:=/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board \
  -p recording_active_topic:=/NS_1/cable_interaction/recording_active \
  -p action_topic:=/NS_1/cable_interaction/action_index \
  -p ee_point_topic:=/NS_1/cable_interaction/ee_point \
  -p scp_destination:=flexcycle@10.157.175.101:/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/motion_correlation_buffers
```

每次录制完成后，时间戳目录会保存在本机，并由后台线程自动复制到 3090 机器。先按第 7 节配置 SSH key。

控制器在两个扰动方向开始时分别记录外力基线。某个方向的相对阻力超过阈值后，机械臂会回到
抓取点并跳过该方向剩余动作，再继续尝试另一个方向。默认阈值为 `6.0 N`，可在控制器启动后调整：

```bash
ros2 param set /NS_1/cartesian_cable_board_interact_controller \
  interaction_resistance_force_threshold 6.0
```

### 3090 终端：启动 CMCor worker

先将本仓库同步到 3090。worker 只在 3090 本地读任务并写结果，不需要连接 Mini PC：

```bash
cd ~/franka_ros2_ws/src/cable_interact

/home/flexcycle/miniconda3/envs/mbest/bin/python -m pointcloud_tools.multigrasp_3090_worker \
  --dataset-root /home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board \
  --cmcor-python-root /home/flexcycle/Motion_Correlation/cmcor
```

worker 会监听：

```text
/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/multigrasp_jobs
```

结果会原子发布到：

```text
/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/multigrasp_result_outbox
```

### Mini PC 终端 3：启动 Multi-grasp orchestrator

```bash
source ~/franka_ros2_ws/install/setup.bash

ros2 run franka_example_controllers multigrasp_orchestrator.py --ros-args \
  -p cable_id:=cable_000 \
  -p initial_grasp_index:=1 \
  -p max_grasps:=3 \
  -p controller_node:=/NS_1/cartesian_cable_board_interact_controller \
  -p recording_active_topic:=/NS_1/cable_interaction/recording_active \
  -p dataset_root:=/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board \
  -p current_grasp_plan_id:=multi_grasp/cable_000/grasp_point_cable_000 \
  -p gpu:=true \
  -p processing_mode:=remote \
  -p remote_worker_host:=flexcycle@10.157.175.101 \
  -p remote_job_inbox:=/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/multigrasp_jobs \
  -p remote_result_outbox:=/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/multigrasp_result_outbox
```

orchestrator 默认也保留 `processing_mode:=local` 单机回退模式。远程模式下默认算法仍为
`correlation`。如需使用 CMCor 的 segmentation 分支，在上面的命令末尾追加：

```bash
  -p algorithm:=segmentation
```

### Mini PC 终端 4：加载控制器、选择方向并触发第一次抓取

确认 Mini PC 三个持续运行的终端和 3090 worker 均正常运行后执行：

```bash
source ~/franka_ros2_ws/install/setup.bash

ros2 run controller_manager spawner \
  cartesian_cable_board_interact_controller \
  --controller-manager /NS_1/controller_manager \
  --controller-manager-timeout 30 && \
ros2 param set /NS_1/cartesian_cable_board_interact_controller \
  target_gripper_direction a && \
ros2 param set /NS_1/cartesian_cable_board_interact_controller \
  target_cable_id "multi_grasp/cable_002/grasp_point_cable_002"
```

上面是一条连续指令：只有控制器成功激活后才设置方向，只有方向设置成功后才设置
`target_cable_id` 并触发运动。将 `target_gripper_direction a` 改成
`target_gripper_direction b` 即可执行候选 `b`；明确指定 `a` 或 `b` 会绕过 joint 7
余量自动选择。最后一个参数设置成功后机器人会立即运动，因此运行前必须先在 RViz 中确认目标
frame、点云和机器人周围环境安全。

第一次抓取完成后，分布式流程会自动：

1. Mini PC 保存 CMCor 序列，并异步上传到 3090。
2. Mini PC orchestrator 向 3090 提交任务 JSON。
3. 3090 worker 等待上传完成，根据运动相关性或 segmentation 结果生成 cable vote。
4. 3090 采样新的抓取点，在本机 outbox 原子发布结果压缩包和 `.ready` 标志。
5. Mini PC 主动从 3090 拉取结果，校验并解包 `grasp_sample_XXX`，将新抓取计划设置到控制器。
6. 控制器自动触发下一次抓取。
7. 达到 `max_grasps` 后停止采样新的抓取点。

例如第一次自动采样后会生成：

```text
pointcloud_tools/info_for_3Dpoint/multi_grasp/cable_000/grasp_sample_001/grasp_sample_001
```

已有该文件并希望从第二次抓取继续时，将终端 3 中的参数改为：

```bash
  -p initial_grasp_index:=2 \
  -p current_grasp_plan_id:=multi_grasp/cable_000/grasp_sample_001/grasp_sample_001
```

然后触发：

```bash
ros2 param set /NS_1/cartesian_cable_board_interact_controller \
  target_cable_id "multi_grasp/cable_000/grasp_sample_001/grasp_sample_001"
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
/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/motion_correlation_buffers
```

每组 Multi-grasp 序列索引保存在：

```text
/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/multigrasp_sequences.json
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

## 7. Mini PC 到 3090 的 SSH

上传在后台线程中执行，不会阻塞 RealSense 图像采集或 ROS 回调。RGB-D 采集脚本会发送完整的
`cable_XXX` 目录；CMCor 录制节点会在序列写入完成后发送完整的时间戳目录。

先在 Mini PC 上配置到 3090 的免密 SSH。自动上传启用了 `BatchMode=yes`，不会等待密码输入：

```bash
ssh-keygen -t ed25519
ssh-copy-id flexcycle@10.157.175.101
ssh flexcycle@10.157.175.101 true
```

当前实现由 Mini PC 主动拉取结果，因此无需配置 3090 到 Mini PC 的反向 SSH。若希望保留反向
登录能力，可在 3090 上额外执行：

```bash
ssh-keygen -t ed25519
ssh-copy-id flexcycle@10.157.174.139
ssh flexcycle@10.157.174.139 true
```

目标目录由 GPU PC 端预先创建。CMCor multi-grasp 使用
`/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board/motion_correlation_buffers`。
未传入 `--scp-destination` 或 `scp_destination`
时，不会执行网络传输。

# 验证 cable 轨迹的操作顺序

下面以 `cable_011` 为例，使用
`cartesian_cable_trajectory_controller` 沿重新计算的 cable 骨架轨迹运动。
当前 controller 会给轨迹的每个点增加 `fr3_link0` 坐标系下的
`Y=+0.02 m` 偏移，使末端在板子上方沿轨迹运动。

执行前确认机器人工作空间内没有人员和障碍物、急停可用，并确认机器人 IP。
设置 `target_cable_id` 会触发真实机械臂立即运动。

## 1. 依次启动机器人、RViz 和轨迹 controller

先在终端 1 中用一条指令启动机器人底层、RViz 和所选 cable 点云发布器：

```bash
source /home/flexcycle/franka_ros2_ws/install/setup.bash
mkdir -p /tmp/ros_logs

ROS_LOG_DIR=/tmp/ros_logs ros2 launch franka_bringup franka.launch.py \
  robot_type:=fr3 \
  namespace:=NS_1 \
  robot_ip:=192.168.3.112 \
  load_gripper:=true & \
rviz2 -d /home/flexcycle/franka_ros2_ws/install/franka_bringup/share/franka_bringup/rviz/cable_interact_realsense.rviz & \
/usr/bin/python3 /home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/cable_on_board/publish_ply_to_rviz.py 011
```

如果机器人的实际 IP 不是 `192.168.3.112`，需要将
`robot_ip` 替换为真实 IP。
最后一个参数 `011` 是要在 RViz 中发布的 cable 点云轨迹 ID；例如要查看
`cable_012`，将它改为 `012`。发布节点会持续运行并发布 `/ply_cloud`，因此
终端 1 需要保持运行。

确认 RViz 已正常显示机器人后，在终端 2 中加载并激活轨迹 controller：

```bash
source /home/flexcycle/franka_ros2_ws/install/setup.bash

ros2 run controller_manager spawner \
  cartesian_cable_trajectory_controller \
  --controller-manager /NS_1/controller_manager \
  --controller-manager-timeout 30
```

等待终端 2 出现：

```text
Controller activated. Holding current pose until target_cable_id is set.
```

## 2. 检查 controller 状态

在终端 3 中执行：

```bash
source /home/flexcycle/franka_ros2_ws/install/setup.bash

ros2 control list_controllers -c /NS_1/controller_manager
```

确认输出中的 `cartesian_cable_trajectory_controller` 状态为 `active`。

## 3. 触发 cable_011 轨迹

保持终端 1、2 继续运行，在终端 3 中执行：

```bash
ros2 param set \
  /NS_1/cartesian_cable_trajectory_controller \
  target_cable_id "'011'"
```

正常情况下，controller 日志会显示：

```text
Received cable 011 skeleton trajectory: 167 points, y offset 0.020 m.
```

随后 controller 会关闭夹爪、移动到轨迹起点附近的悬停位置，再沿jia zai
`cable_011/skeleton_from_pc` 轨迹运动。

注意：当前悬停偏移 `kHoverApproachOffset` 为 `Y=-0.15 m`。在新的 setup
中，`fr3_link0 +Y` 指向板子上方，因此运行前必须确认这个悬停方向不会使
机器人朝板面运动。
