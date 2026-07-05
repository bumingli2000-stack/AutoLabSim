# AutoLabSim 更新记录

这个文件用于记录 AutoLabSim 每次阶段性修改的内容、原因和验证结果。后续同步到 GitHub 后，可以直接在仓库里查看完整演进记录。

## 2026-07-05

### 开始构建移液任务场景，物体位置已经调整好，但是当前还不能交互  --LCY



## 2026-07-04

### 双臂旋拧离心管盖任务重构

- 将双臂开盖任务的主要参数集中到 `autolabsim/tasks/screw_cap.py`，减少命令行参数和任务内部默认值之间的冲突。
- 精简 `generate_tube_grasp_batch.py` 的 CLI，只保留场景、任务、采集数量、图像记录等采集入口参数。
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
- 使用 `scripts/generate_tube_grasp_batch.py --task tube_then_cap_grasp` 生成测试 episode。
- 验证结果：完整任务可以完成，`released=True`，瓶盖成功旋拧释放，离心管可放回试管架，最后双臂回初始姿态。

### 后续注意

- 如果末端轨迹仍然绕弯，优先调小 `cartesian_step_size`，而不是直接调 TOPP 参数。
- 如果瓶盖夹持仍有穿模，优先调 `cap_close_gripper`。
- 如果管身固定不牢，优先调 `tube_close_gripper` 或管身抓取高度 `tube_grasp_height`。
- 后续新增任务时，建议继续沿用“任务参数集中在 task config，CLI 只负责采集入口”的组织方式。
