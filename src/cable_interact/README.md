# Cable 交互任务

当前单双臂 MoveIt demo 位于 [cable_moveit_grasp](cable_moveit_grasp/README.md)。
双臂操作说明：[dual_arm_automatic.md](cable_moveit_grasp/docs/dual_arm_automatic.md)。

实时点云采集使用独立源码 `/home/flexcycle/Motion_Correlation/cmcor/pointcloud`，入口为 `ros2 launch cable_pointcloud mini_pc.launch.py viewer:=false`。
活动标定位于 `cable_moveit_grasp/config/calibration/fr3_calibration_apriltag_depth_003.calib`，单双臂 launch 共用此文件。

旧离线点云工具、数据及其专用线缆控制器已移除；不再安装旧 multigrasp 程序。AprilTag 和其他通用控制器保持独立。
