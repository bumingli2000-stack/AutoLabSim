# AutoLabSim 更新记录

这个文件用于记录 AutoLabSim 每次阶段性修改的内容、原因和验证结果。后续同步到 GitHub 后，可以直接在仓库里查看完整演进记录。
## 2026-07-17
优化了电动移液枪的装枪头的流程，放弃了运动学插值安装插头，下压过程采用纯动力学接近枪头

## 2026-07-15
将任务一的数据生成后用于ACT训练，训练结果已经部署到环境中，但结果很不理想，可能的问题如下：
1.机械臂控制接口定位存在误差，目前采用是一个PD控制器去做插值，需要调整或重写。
2.ACT本身的定位精度达不到数据生成时的程度，尤其在数据具有一定分布的时候，原有的瓶盖交互逻辑写的过于生硬，与夹爪的交互不够，可能需要修改交互逻辑、引入更多的物理交互。

3.是否需要更换到其他仿真平台，比如maniskill、isaac_sim之类的？

## 2026-07-11
任务一 screw_cap 和任务二 pipette_grasp 都已经完成重构，目前文件结构如下（以任务二为例）：
pipette_grasp.py   任务的主体流程，包括各个模块的定义及各个流程的顺序执行
pipette_targets.py 用于定义任务目标点，并进行坐标系转换与IK规划
pipette_scene.py   用于查询任务场景，比如解析枪头的实时位置，移液枪的位置等
pipette_metadata.py用于构造该任务的数据，在本任务中，采集了较为完整的数据，可以根据需要转换成不同格式

文件入口：
generate_pipette_grasp_batch.py中的定义了一组基本参数，传入autolabsim.tasks.cli文件的main函数-->run_batch()-->run_episode()
-->__init__.py中的create_task()-->注册表TASK_REGISTRY-->_create_pipette_grasp_task()-->pipette_grasp.py中的PipetteGraspTask 完成对象创建
-->之后在run_episode()中执行任务主体流程


## 2026-07-09
完善了部分文件注释，修复了一些无用逻辑，合并了本源关于scene_mujoco.xml文件的调整
推荐关于任务二的阅读逻辑
先看 pipette_grasp.py 的 run()
→ 再看对应的 _stage_*()
→ 再跳到 pipette_targets.py 看目标如何定义
→ 场景对象不清楚时看 pipette_scene.py
→ 最后看 metadata 记录了什么

## 2026-07-08
## 完成架构重构，核心路线已经从“任务类里手写坐标转换/IK/执行”改成：
## 选择对象 -> 定义 TaskTarget -> TaskTargetPlanner 规划 -> TaskTargetExecutor 执行 -> 更新 attachment
重构了整个文件的任务规划逻辑，新增TaskTarget类，该类主要用于描述用于规划路径的目标点，将目标点的各种位置、状态、操作等属性都封装在其中，并基于此类重构了规划、执行等接口。基于此类实现了对规划和操作任务的高度抽象，后续拓展到新任务场景时，只需要调整目标点的属性，其余接口变动较小。
目前版本还未推广到其他任务，稳定性还有待测试                       --LCY

pipette_grasp文件被分解为三个部分，autolabsim/tasks/pipette_targets.py用于设置各个规划点的属性，autolabsim/tasks/pipette_scene.py用于获取场景中任务相关物体的位姿，autolabsim/tasks/pipette_metadata.py用于记录实验结果，autolabsim/tasks/pipette_grasp.py用于执行规划操作

修改文件
[task_target.py](/autolabsim/task_target.py)：新增 PoseOffset、TaskTarget.target_offset、offset_local()、with_approach_offset()。
[motion_context.py](/autolabsim/motion_context.py)：新增 PlanningContext、ExecutionContext、KinematicBinding、SiteAttachment 等类型。
[planner.py](/autolabsim/planner.py)：新增通用 TaskTargetPlanner，支持一级/多级 attachment 链。
[executor.py](/autolabsim/executor.py)：新增通用 TaskTargetExecutor，接管插值、夹爪时序、settle、visual servo、约束维持、误差记录。
[pipette_grasp.py](/autolabsim/tasks/pipette_grasp.py)：迁移到 Planner/Executor，删除 pipette 任务内通用运动执行逻辑。
[tasks/__init__.py](/autolabsim/tasks/__init__.py)：迁移 pipette 配置构造为嵌套配置。
[tasks/cli.py](/autolabsim/tasks/cli.py)：删除无运行端接收者的 --grasp-offset、--tool-roll。
[visualize_task_points.py](/scripts/visualize_task_points.py)：不再依赖已删除的 pipette lift_offset 配置。
[__init__.py](/autolabsim/__init__.py)：导出新 motion/planner/executor 类型。
[test_motion_architecture.py](/tests/test_motion_architecture.py)：新增 8 个纯 Python 架构测试。


## 2026-07-05

### 开始构建移液任务场景，物体位置已经调整好，但是当前还不能交互  --LCY



## 2026-07-04

### 双臂旋拧离心管盖任务重构

- 将双臂开盖任务的主要参数集中到 `autolabsim/tasks/screw_cap.py`，减少命令行参数和任务内部默认值之间的冲突。
- 精简批量生成 CLI，只保留场景、任务、采集数量、图像记录等采集入口参数。
- 完善任务流程：抓瓶盖、向上提起离心管、另一只机械臂横向夹住离心管、棘轮式旋拧瓶盖、放置瓶盖、放回离心管、双臂回初始位姿。
- 恢复并明确瓶盖提起阶段，使用 `cap_post_offset` 控制抓住瓶盖后向上提起离心管的高度。
- 新增 `return_home_after_task` 和 `return_home_steps`，任务结束后两个机械臂会回到 reset 后的初始姿态。

### 轨迹规划与精度

- 引入 TOPPRA 轨迹时间参数化，新增 `autolabsim/topp.py`，并在 `requirements.txt` 中加入 `toppra`。
- 将普通位姿移动阶段改成“笛卡尔密集路点 + 连续 IK + TOPP 时间参数化”，避免只用稀疏关节路点导致末端轨迹绕弯。
- 新增统一插值参数 `cartesian_step_size` 和 `cartesian_min_steps`，用于控制普通位姿阶段的末端笛卡尔插值密度。
- 收紧 IK 位置和姿态容差，并增加 waypoint settle 逻辑，用于提高夹爪 `pinch` 到目标抓取点的定位精度。
- 调整 `cap_tool_roll`，避免瓶盖机械臂在抓盖前出现不必要的腕部大角度翻转。

### 夹爪与碰撞

- 定位到瓶盖夹爪碰撞的主要原因是抓取定位误差叠加夹爪闭合过紧。
- 将夹爪闭合值拆分为 `cap_close_gripper` 和 `tube_close_gripper`：瓶盖轻夹，管身强夹。
- 保留 `close_gripper` 作为通用旧参数，但瓶盖和管身任务阶段分别使用更具体的闭合参数。
- 新增 `scripts/dump_episode_contacts.py`，可以回放 episode 并按 phase 输出 MuJoCo contact pair，辅助定位具体是哪两个 geom 发生碰撞。

### 可视化与调试工具

- 新增 `scripts/visualize_planned_trajectory.py`，用于可视化实际规划出来的夹爪末端轨迹。
- 扩展 `scripts/visualize_task_points.py`，支持当前双臂开盖任务的抓取点可视化。
- 扩展 `scripts/view_scene.py`，支持显示相机、接触点、geom group、site group 等调试视图。
- 增加 site error 日志，运行任务时会打印关键阶段目标点和实际 `pinch` 位置之间的误差。

### 场景和配置

- 调整 `model/scenes/scene_mujoco_fast_tubes.xml`，围绕当前 50ml 离心管旋拧任务优化场景。
- 更新 reset 配置，使 50ml 离心管可以按任务需求放置到试管架槽位。
- 调整 2F85 夹爪相关模型/碰撞设置，用于更好地观察和调试夹持行为。

### 验证

- 使用 `python -m py_compile` 检查了核心 Python 文件。
- 使用 `scripts/generate_task_batch.py --task tube_then_cap_grasp` 生成测试 episode。
- 验证结果：完整任务可以完成，`released=True`，瓶盖成功旋拧释放，离心管可放回试管架，最后双臂回初始姿态。

### 后续注意

- 如果末端轨迹仍然绕弯，优先调小 `cartesian_step_size`，而不是直接调 TOPP 参数。
- 如果瓶盖夹持仍有穿模，优先调 `cap_close_gripper`。
- 如果管身固定不牢，优先调 `tube_close_gripper` 或管身抓取高度 `tube_grasp_height`。
- 后续新增任务时，建议继续沿用“任务参数集中在 task config，CLI 只负责采集入口”的组织方式。
