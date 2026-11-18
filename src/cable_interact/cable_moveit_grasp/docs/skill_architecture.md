# 双臂技能结构与 FIX 投影路径选择

当前使用共享技能后端，原有自动状态机顺序、五轮计数和人工选点回退保持不变。
技能是同一 C++ 节点的独立文件，不是各自连接硬件的进程；两臂运动串行。

| 文件 | 作用 |
|---|---|
| `scripts/alternate_grasp_state_machine.py` | 原状态转换、循环、失败保持 |
| `scripts/cable_moveit_alternate_grasp.py` | 服务编排、暂停/恢复、人工回退、JSON契约校验 |
| `skills/*.json` | 技能参数、前置条件、成功条件、资源与当前任务描述 |
| `src/gripper_collision.cpp` | 每次夹爪动作前读取两臂与两手新鲜实测状态，检查完整手指开合扫掠与世界/自碰撞/双臂碰撞 |
| `src/dual_arm_motion_node.cpp` | 节点入口 |
| `src/dual_motion_backend.cpp` | 公共服务、实测状态、场景、碰撞、轨迹执行与终点投影复核 |
| `src/skills/home_skill.cpp` | 录制关节 HOME；组合命令先调用开爪 |
| `src/skills/grasp_skill.cpp` | 原候选轨迹抓取与夹爪成功记录 |
| `src/skills/fix_skill.cpp` | 录制位置目标；优先保持当前朝向，必要时比较备用朝向的笛卡尔路径 |
| `src/skills/release_skill.cpp` | 开爪并更新持线记录 |
| `src/skills/verify_skill.cpp` | 验证抓取成功记录，不含滑移检测 |
| `src/grasp_candidate_planner.cpp` | 抓取候选生成与预览，不执行物理动作 |
| `include/cable_moveit_grasp/fix_projection.hpp` | 目标正反朝向、TCP-Y XZ投影有向累计、终点绝对值判据 |
| `include/cable_moveit_grasp/fix_path_shortcut.hpp` | 碰撞检查支持的简化，只以最终累加角判断是否合格 |

## 当前 FIX 流程

`fixA/fixB` 七个录制关节角通过 FK 只提取 TCP 目标位置，录制朝向不参与候选生成。
先保持本次 FIX 开始时的 TCP 朝向，尝试直线平移；完整、无碰撞且投影判据合格就直接执行。
失败后，围绕起始 TCP 的本地 X/Y/Z 轴分别旋转 ±30/60/90/120/150°及180°，生成33个备用朝向。
每个备用朝向尝试直接、先平移再旋转、先旋转再平移三种笛卡尔路线，最多100次路径请求。
备用路径按最终投影累加角绝对值较小择优，接近相同时按关节总行程较小择优。
只使用笛卡尔插值，无RRT后备搜索，也不以关节空间捷径替换笛卡尔路线。
按关节步长0.01 rad密集复核双臂、自碰撞、桌板、相机、关节范围与状态；目标位置误差<=1mm。
朝向候选仅用于规划，不要求到达录制朝向。执行保存的原轨迹，执行前重新校验，并保留HOME/FIX关节到位检查。
如果起始位置已在1mm以内，不改变朝向、不发运动。

## 投影判据

各点TCP +Y轴表达在本臂固定link0（A为a_fr3_link0，B为fr3_link0）下，取XZ投影并归一化。
相邻单位投影u、v的有向增量为 `atan2(u_z*v_x-u_x*v_z, u_x*v_x+u_z*v_z)`；正方向为绕link0 +Y右手方向。
沿路径相加，不对最终结果取模，只要求 **abs(S_N)<=180°**。
沿途峰值不作拒绝条件。+200°再退回+60°可以合格；最终-63.22°可以合格；最终116.78°合格、296.78°拒绝。
投影模长<1e-6时角度未定义，明确拒绝。

配置：

```yaml
fix_tcp_max_total_rotation: 3.141592653589793
```

超限路径发布到 `/dual_arm/dual_arm_motion/rejected_fix_a` 或 `rejected_fix_b`，
RViz勾选 `A/B FIX Rejected Preview ONLY` 观察。预览不是执行授权。
日志 `FIX qualified candidate=...` 列出合格路径，`FIX selected candidate=...` 说明最终选择。
失败则保持夹持，没有取消投影判据的备用执行路径。

## 已知边界与回退

FIX 只尝试有限组朝向的笛卡尔路线，不会通过随机采样搜索绕障路径。HOME 与抓取候选仍使用原有 MoveIt 规划。
找到296°长路径并不证明不存在-64°短路径，也不能直接减360°后放行。
已有实际抓取/轨迹控制仍依赖MoveIt与ros2_control，FCI通信故障须另外排查。

修改前反馈版已备份至 `/tmp/cable_moveit_grasp_feedback_before_projection_restore_20261109.tar.gz`，
恢复说明清单位于 `/tmp/cable_moveit_grasp_projection_restore_manifest_20261109.json`。
恢复反馈版需替换相应源码/配置后重新编译并重启dual，不可只把运行参数切回反馈版。
本次只编译和离线验证，未发送真实运动指令。

抓取重试与回HOME后重选点流程见 `dual_arm_automatic.md` 的“抓空自动恢复”；该恢复成功后才向状态机返回抓取成功，恢复耗尽后在原阶段等待人工新点。

普通自动选点使用 `continuous_cable_selection.hpp`：B动态X范围、A向+X扫描30cm、至少5cm连通支持和局部PCA质量；详见运行文档“普通自动选点：连续点云质量”。
