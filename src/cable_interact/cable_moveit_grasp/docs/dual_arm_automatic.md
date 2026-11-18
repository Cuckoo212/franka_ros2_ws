# 双臂自动交替抓取

双臂 launch 默认启动 `cable_moveit_alternate_grasp`，节点初始为 idle，不发送运动。
相机、GPU 线缆点云、标定转换、候选规划、机器人驱动仍由原流程提供。
单臂抓取入口不变；原六个手动命令保留，自动流程使用独立的 `/auto/` 动作服务。

## 启动与首次抓取

当前推荐使用已验证正常的方式：**独立启动 A/B gripper server，dual launch 使用 `start_grippers:=false`**。
相机和 GPU 点云程序沿用原来的独立终端；相机入口默认不打开 RViz。
处理好上一轮线缆，双臂从空载开始。各夹爪只保留一个服务器实例，停止旧的重复实例后按以下顺序启动。

### 终端 1：B（右臂）夹爪服务器

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash
ros2 launch franka_gripper gripper.launch.py \
  robot_ip:=192.168.3.101 namespace:=/NS_1 \
  robot_type:=fr3 use_fake_hardware:=false
```

### 终端 2：A（左臂）夹爪服务器

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash
ros2 launch franka_gripper gripper.launch.py \
  robot_ip:=192.168.3.102 namespace:=/NS_2 arm_prefix:=a \
  robot_type:=fr3 use_fake_hardware:=false
```

两个终端分别出现 `Connected to gripper` 后保持运行。

### 终端 3：双臂驱动、MoveIt、RViz 和空闲状态机

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash
ros2 launch cable_moveit_grasp dual_arm.launch.py \
  use_fake_hardware:=false execution_enabled:=true \
  start_grippers:=false start_coordinator:=true \
  motion_config:=/home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/config/dual_arm_motion.yaml
```

`start_grippers:=false` 让双臂入口使用前两个终端的服务器，不重复建立夹爪连接；
hand/TCP 模型和原 grasp 参数均保留。不要在此模式下再启用另一份内嵌夹爪服务。

### 终端 4：启动五轮自动流程

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash
alternate_start
alternate_state
```

状态中 `max_cycles: 5` 表示默认五轮上限。首次回 HOME 和 B first 人工选点只执行一次。

`alternate_start` 开始执行真实动作：A 先打开夹爪并回 HOME_A，成功后 B 打开夹爪并回 HOME_B。
随后状态为 `waiting_manual_point`，`waiting_arm: b`、`ready_for_manual_point: true`。
此时在 RViz 用 Interact 点击 B RIGHT，再用 Publish Point 点击首次线缆抓取点。
首次 B 强制使用这个新人工点，不使用固定 Y 自动选点。

**一旦生成有效候选轨迹，状态机自动执行首次 B 抓取，无需再次运行 grasp_b 或手动确认。**
旧的 RViz 预览不生效：每次等待都清空旧选择，要求点选后的候选版本高于等待基线。
若手动点选也无法生成候选，继续等待，可重新点选。

## 完整执行顺序

1. HOME_A → HOME_B。
2. 等待 B 的首次人工抓取点。
3. B 抓取成功 → FIX_B。
4. A 在B fr3_link0下，从B当前TCP X+0.10m到X+0.40m，按X递增寻找连续至少5cm且局部切线可靠的cable段选点抓取。
5. A 抓取成功 → B 打开夹爪并回 HOME_B。
6. B 到 HOME_B 后 → A 持线移动到 FIX_A。
7. B 在B fr3_link0下，从X=-0.30m到A当前TCP在B坐标中的X-0.10m，按X递增寻找连续至少5cm、局部切线可靠且距桌板模型表面大于3cm的点抓取。
8. B 抓取成功 → A 打开夹爪并回 HOME_A。
9. 未满五轮时回到第 3 步的 FIX_B，再依次执行第 4～8 步。此时 B 已持线，不重复执行首次 B 抓取。

每轮 B 抓取成功且 A 回到 HOME_A 后，B 都必须先持线移动到 FIX_B，成功后才进入 A 的自动抓取。

**完成一次第 8 步计为一轮，默认累计五轮。** 初始 HOME/first 选点不另计一轮。
第五次 A 回 HOME 成功后状态变为 `mode: completed`、`completed_cycles: 5`，
保留本轮已完成的 `a_release_and_home` 状态，不进入第六次 FIX_B。
此时 A 已松爪在 HOME_A，B 仍夹持电线；不会额外松开 B 或自动回 HOME_B。
失败的动作不计为已完成一轮，恢复后继续原来的计数。

机械臂动作严格串行；HOME/FIX 的成功到位和抓取内部校验完成后才进入下一动作。
抓取成功由轨迹控制器执行成功、Franka Grasp Action 成功及内部夹持记录共同决定，
没有额外加入之前移除的抓取 TCP/关节到位门槛，也没有独立的视觉滑移检测。
FIX 为录制及翻转两个完整目标规划并比较终点投影累加角；保留碰撞和关节范围检查，参数及边界见技能架构文档。
抓取前默认等 0.5 秒供移动后的点云刷新，不需要人工观察确认。

## 没有自动候选

`grasp_a/grasp_b` 的选点、PCA/IK 或轨迹规划失败，并且未发送任何物理动作时：

- 状态机自动切到 `waiting_manual_point`，自动选择对应 A LEFT / B RIGHT。
- 查看 `alternate_state` 的 `waiting_arm`，等待 `ready_for_manual_point: true`。
- 在 RViz 重新 Publish Point。有效候选准备好后立即继续抓取和后续状态。
- **自动模式中不要再在终端运行 grasp_a/grasp_b**，它们属于外部手动接管，会暂停状态机。

缺少或过期的机器人关节状态、HOME 位置不符合要求等状态错误不属于选点问题，
会进入 failed_hold，修复后使用 resume。规划器的详细失败原因由 backend 传回状态机。

## 抓取/运动失败与人工恢复

轨迹执行或夹爪动作失败时进入 `failed_hold`，不继续交接，不自动松开另一臂。
故障处在原步骤，保留已有夹持记录；不会因为服务超时或未确认结果而自动重试动作。

例如 A 抓取失败，另一个终端：

```bash
source /home/flexcycle/franka_ros2_ws/src/cable_interact/cable_moveit_grasp/scripts/robot_env.bash
home_a
# 等待 success=True，确认手动松爪并回位完成。
alternate_resume
```

B 抓取失败则使用 `home_b`，再 `alternate_resume`。
手动 HOME 允许越过对应臂的故障锁，先松爪再按原到位检查返回 HOME；
只有失败臂的 HOME 成功才清除其后端故障。恢复失败不会清除故障。
自动调度的 HOME 不具备这个越过故障的权限，也不能释放唯一已确认持线臂。

`resume` 会重新查询后端实际夹持/HOME 记录。抓取重试必须本臂回 HOME、
另一臂仍持线（首次 B 例外）。如果这些条件不成立，会拒绝恢复并返回原因，
不会推断手动动作已完成状态机步骤。首次 B 的恢复会重新等待人工 first 点。
FIX 失败若人工 HOME 释放了本臂，必须先手动恢复该臂的成功抓取记录，
或双臂空载回 HOME 后 reset/start 重开一轮，不能直接恢复一个已经不持线的 FIX。

## 暂停、查看、重开

```bash
alternate_pause   # 只停止后续调度；已提交的动作仍被跟踪，不是机器人运动急停。
alternate_state   # 读取一次 phase/mode/原因/循环数/当前请求。
alternate_resume # 检查后端记录后继续。
alternate_reset  # 必须暂停、无在途请求、双臂空载在 HOME、无后端故障，重置到 idle。
```

原六个手动命令都会通知状态机暂停调度，不会悄悄和自动流程交错。
恢复后须显式 resume，防止人工处理线缆过程中自动进入下一步。
状态机超时会保留在途请求并停止推进；迟到的成功响应不会直接触发下一动作。
未解决在途动作前不允许 resume/reset。`pause`/`stop` 服务不取消真实在途轨迹。
单纯节点/驱动重启不会恢复夹持记录，也不会自动重启流程。

服务前缀 `/cable_moveit_alternate_grasp`，提供 start、pause、stop、resume、reset，均为 Trigger。
状态话题 `/cable_moveit_alternate_grasp/state` 为保留最新一帧的 JSON String。
原 execute_current_step、complete_step 的逐步确认入口不再用于新自动状态机。

配置 `config/alternate_grasp.yaml`：

```yaml
physical_backend: false  # dual launch 用 execution_enabled 参数覆盖。
backend_timeout: 240.0
perception_settle_time: 0.5
max_cycles: 5  # 默认五轮；每次第 8 步完成计一轮，第五轮后停止，B 仍持线。
# 显式设为 0 才持续循环。
```

不需要状态机时可 `start_coordinator:=false`，继续使用原六指令工作流。

## 验证

- `test_alternate_state_machine.py`：实际顺序、循环、失败不推进、恢复夹持条件。
- `test_coordinator_backend.py`：旧预览拒绝、暂停中的完成、未知/迟到结果、选点/物理失败分类。
- `manual_automatic_sequence_smoke.py`：隔离 ROS 94，仅模拟六动作服务，完整两轮及注入失败、人工 HOME + resume。
- `manual_home_recovery_smoke.py`：隔离 ROS 93，真实 MoveIt/规划节点配模拟 Action 服务器，首次人工计划握手、抓取失败锁定及手动 HOME 恢复。

上述测试不启动真实机械臂驱动、不连接真实硬件。

## 夹爪启动隔离与对照

双臂入口现在直接分别包含标准 `franka_gripper/gripper.launch.py`，参数与独立启动相同，
并关闭 arm bringup 内嵌的服务器。每臂各一实例，hand/TCP 模型不变。
launch 参数仍支持 `start_grippers:=true` 的集成启动；本文推荐命令明确使用
`start_grippers:=false`，完全使用外部已启动的服务器。
独立的 arm bringup 默认 `start_gripper_server:=true`，保留原行为。

若独立 B server 工作正常，先保持该服务器，再以 `start_grippers:=false` 启动双臂入口，
不要同时启动另一份同 IP 的夹爪服务。此时不要先运行 alternate_start，
先在机械臂静止、同一物体位置及相同初始开口条件下，用相同 Grasp Action 做对照。
完整自动流程还需要 A 的独立 server（.102、/NS_2、arm_prefix=a）。

这个开关用于隔离连接与启动问题；单独成功、自动抓取失败尚不能证明启动参数有错，
也可能是自动轨迹后的接触位置、姿态或通信状态不同。目标参数保持不变，
用新增 Grasp goal/result 日志比较实际收到的命令和硬件反馈。

## 相机碰撞保护

双臂 launch 读取同一个 `calibration_file`，将 RealSense 保护盒加入共享规划场景，物体 ID 为 `realsense_camera`，坐标系为 B 的 `fr3_link0`。与点云完全共用光学坐标转换，A/B 的抓取、HOME、FIX 均检查该场景；执行前若保护盒缺失会拒绝轨迹。

在 `config/dual_arm_motion.yaml` 配置：

```yaml
camera_obstacle_enabled: true
camera_obstacle_size: [0.12, 0.05, 0.04]
camera_obstacle_center_optical: [0.0, 0.0, 0.0]
```

尺寸单位为米，顺序为光学坐标 X（右）、Y（下）、Z（前）。中心偏移也使用光学坐标。相机型号为 D456；按用户给出的顺序，保护盒 X/Y/Z 尺寸设置为 12/5/4 cm，中心仍为标定光学中心。这是用户指定的近似范围，并非已核实的机身尺寸；需在 RViz 中核对其覆盖机身、支架及标定误差，必要时增大尺寸或调整中心偏移。

修改后重启 dual launch，启动命令保持不变（独立 A/B gripper server 继续运行，dual 使用 `start_grippers:=false`）。在 RViz 的 Obstacles 中核对相机保护盒。相机移动后必须重新标定。碰撞物体未加入任何碰撞豁免。

## 当前 FIX 与技能框架

FIX 只使用录制TCP位置，优先保持当前朝向直线移动；失败时比较备用朝向的笛卡尔路径，要求实际路径最终TCP-Y在本臂link0的XZ投影有向累加角绝对值<=180°。
不再使用位置反馈、起始朝向保持或沿途峰值限制。
详细代码结构、参数、控制边界见 [skill_architecture.md](skill_architecture.md)。


## 抓空自动恢复

A/B 在候选轨迹成功后闭爪。只有明确结束、success=false且error为空的夹爪动作（ABORTED或SUCCEEDED返回码）才作为可重试抓空；超时、取消、设备报错、通信/运动失败仍停止并要求人工恢复。
同一点依次尝试原始TCP、原始TCP本地-Z 1cm、原始TCP本地+Z 1cm（即从第二次位置向+Z移动2cm）。朝向保持原始抓取朝向；这里不是link0的Z轴。
两个偏移动作均先开爪，再用1mm步长笛卡尔规划与完整碰撞校验，成功执行后再闭爪。
三次都失败则开爪，失败臂回HOME；另一臂不松爪。HOME成功后从最新点云重新生成候选与整条抓取轨迹：
- A：B当前TCP在B fr3_link0下的X+0.15m，容差±0.01m。
- B：刚才交付执行的点云点在B fr3_link0下的X+0.05m，容差±0.01m；不使用普通连续段遍历规则选择这次恢复点。
X带内仍取距三维参考点最近的点，不是线缆弧长。
默认grasp_max_reselections=1：最多两个点、六次闭爪。所有次数失败后开爪回HOME，自动状态机停在原抓取阶段等待新RViz人工选点；新点成功才继续交接。
若恢复选点无候选，也在HOME等待人工新点。手动grasp_a/b会返回失败提示，不自动推进状态机。
普通A自动选点从B当前TCP +X 0.10m开始向+X搜索0.30m；0.15m仅用于抓空后的恢复选点。


## 夹爪开合碰撞检查

所有后端发出的Move/Grasp action在发出前统一检查手指扫掠，不只检查目标宽度。
读取两臂14个关节和两手4个手指的原始带时间戳测量（<=1秒），不以默认手指位置代替；缺少任一夹爪测量则拒绝动作。
Move按当前开度到目标开度插值；Grasp保守检查到完全闭合（抓空时可能越过请求宽度）。每根手指步长<=0.5mm。
保持其余机器人姿态，使用最新共享场景对整个双臂机器人检查自碰撞、臂间碰撞和世界障碍，保留原SRDF允许碰撞设置。
桌板与启用的相机对象缺失、状态无效或任一采样点碰撞都会在发送夹爪goal前抛错，日志列出首个碰撞物体对。
该检查是动作前离散检查，不是执行中的连续碰撞监控；电线点云仍不作为世界碰撞物体。
接近零点的手指测量误差仅在0.1mm以内夹到模型硬限位，较大越界直接拒绝。


## 普通自动选点：连续点云质量

所有X及当前TCP坐标统一表达在B的fr3_link0，A的TCP也先转换到B坐标，不能直接使用A本地link0的X。
B首个抓取仍人工选点；后续B普通选点不再按固定Y或HOME距离，而扫描[-0.30, A当前TCP的X-0.10]m。
A普通选点扫描[B当前TCP的X+0.10, B当前TCP的X+0.40]m。
在区间内按原始点X升序，选第一个通过质量检查的点，不把没有点云的X位置当成抓取点。
默认以2mm体素代表点构建3D邻接图；相邻代表点距离<=6mm才连通。有效连通段须有两个实测代表点相距>=5cm，因此不会通过累加噪声点/分支边长度伪造5cm。
这是连续长度的保守下界，不是严格的线缆中心线弧长；弯曲成小圈、空间跨度不足5cm的段可能被拒绝。允许小于等于6mm的离散采样间隔，大于此值即断段。
还沿用局部半径2.5cm、至少12个点和PCA切线质量检查；第一个点不满足时继续向更大X遍历。
连续段支持点限定在搜索X区间内；局部切线估计仍使用完整点云邻域。没有合格段、B搜索上界小于下界或后续抓取轨迹规划失败，仍回退到RViz人工选点。
抓空后回HOME的A +X15cm、B原点+X5cm恢复规则不变；本次连续质量筛选只作用于普通自动选点，不改变人工选点。


B普通自动选点要求桌板模型表面间距严格大于 `grasp_b_min_board_clearance`（默认0.03m）；等于3cm也跳过。若姿态生成的近桌板阈值更大，选点取两者较大值。原姿态生成阈值 `near_board_height_threshold=0.01m` 不变。
间距复用原 `boardHeightGap` 计算，沿B的+Y衡量，不是直接比较点的Z。该过滤作用在X顺序遍历中；不合格则继续向更大的X搜索，不改变5cm连续段支持点。
通过过滤后不会触发两候选近桌板规则，而使用常规24姿态采样。A、首次B人工选点和抓空恢复选点保持原有规则。
如果所有质量合格点都贴近桌板，回退人工选点，不自动放宽阈值。桌板模型未启用时也拒绝这一路自动选点。
