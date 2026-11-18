# Agentic dual-arm cable skills

本包只维护双臂技能执行路径。原始单臂执行和历史实验入口已移除；当前 FIX 已恢复为投影路径选择。

- A：左臂 `192.168.3.102`，`/NS_2`；B：右臂 `192.168.3.101`，`/NS_1`。
- 原自动状态机负责五轮交替、首次人工 B 选点、自动选点回退和失败保持。
- HOME / GRASP / FIX / RELEASE / VERIFY 在 `src/skills/` 独立实现，共享一个执行节点。
- `skills/*.json` 描述技能和当前任务；当前状态转换仍由确定性状态机实现。
- `grasp_candidate_planner` 只生成和预览轨迹，不执行运动或控制夹爪。
- FIX 为录制及翻转两个完整目标分别规划，比较实际路径终点TCP-Y XZ投影有向累加角，绝对值<=90°才合格。

## 目录

```text
launch/dual_arm.launch.py       唯一主启动入口
src/dual_arm_motion_node.cpp    节点入口
src/dual_motion_backend.cpp     共享状态、服务、故障、场景及执行
src/skills/                    C++ 动作技能
src/grasp_candidate_planner.cpp 抓取候选规划与预览
include/cable_moveit_grasp/     当前公共模型/几何/校验工具
scripts/                       状态机、JSON校验、标定、点云和驱动准备
skills/                        技能及任务JSON
config/                        双臂控制、FIX投影、候选规划及RViz配置
test/                          当前离线回归测试
docs/                          当前运行步骤与架构说明
```

运行步骤见 [docs/dual_arm_automatic.md](docs/dual_arm_automatic.md)，文件职责和反馈边界见
[docs/skill_architecture.md](docs/skill_architecture.md)。

## 构建与测试

```bash
cd /home/flexcycle/franka_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --packages-select cable_moveit_grasp
colcon test --packages-select cable_moveit_grasp
colcon test-result --test-result-base build/cable_moveit_grasp --verbose
```

## 运行

先按运行文档分别启动A/B独立 gripper server，并启动点云发布。随后：

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash
ros2 launch cable_moveit_grasp dual_arm.launch.py \
  use_fake_hardware:=false execution_enabled:=true start_grippers:=false \
  motion_config:=/home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/config/dual_arm_motion.yaml
```

另一个终端加载同一环境后运行 `alternate_start` / `alternate_state`。
手动保留 `home_a home_b grasp_a grasp_b fix_a fix_b` 六条命令。
真实闭环控制需在实际设备上验证；离线测试不代表实际执行效果已验证。
