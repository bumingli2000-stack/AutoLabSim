# AutoLabSim 整体工程文件架构

1. 场景模型层
[scenes/scene_mujoco.xml]
原始主场景入口，定义双臂、桌子、离心架、相机、物体等。
[scenes/scene_mujoco_fast_tubes.xml]
为数据生成准备的快速版场景，主要是碰撞更轻，跑得更稳。

2. 子模型 / 资产层
被主场景引用的零件模型。
[models/grippers/2f85.xml]
正常夹爪模型。
[models/grippers/2f85_fast.xml]
轻量碰撞版夹爪模型。
assets/ 下面的 robot/, container/, rack/, tool/, incubator/, workbench/
放 mesh、obj、stl 这些几何资产。

3. 场景布局配置 / reset 层
这层决定“每次 episode 开始时，东西摆在哪、哪些物体激活、哪些隐藏”。
[reset_default.json]
比较通用的默认初始状态。
[reset_single_tube_random.json]
当前单管任务用的配置。它会指定：机器人初始关节
哪根管子是 active
cap 跟 tube 的配对关系
tube 随机落在哪个 slot
其他 tube/cap 怎么隐藏

4. 任务逻辑层
这层决定“机器人要做什么”。
[autolabsim/tasks/tube_grasp.py]
单臂抓管任务。
[autolabsim/tasks/screw_cap.py]
双臂旋拧开盖任务。
[autolabsim/screw.py]
旋拧过程的专门逻辑。
[autolabsim/tasks/__init__.py]
任务注册表，负责按名字创建任务。

5. 仿真运行层
这层负责“把场景跑起来”。
[autolabsim/mujoco_env.py]
MuJoCo 环境封装。
[autolabsim/scene.py]
读取/修改 joint、site、body 这些场景对象。
[autolabsim/reset_config.py]
读取并应用 reset 布局配置。
[autolabsim/scene_profile.py]
统一管理“这个任务默认用哪个 scene、哪个 reset、哪些 camera”。

6. 数据记录与脚本层
这层负责生成、查看和导出数据。
[scripts/generate_tube_grasp_batch.py]
批量生成 episode 的入口。
[scripts/visualize_task_points.py]
可视化抓取点/预抓点/抬起点。
[scripts/view_episode.py]
回放已保存的 episode。
[autolabsim/recorder.py]
记录 qpos/qvel/ctrl/image。
[autolabsim/episode_io.py]
存取 episode。

## Reset 配置模板

如果你要新增一个任务场景，通常不需要先改 task 代码，第一步往往是先新建一个 reset 配置文件。建议直接复制：

```bash
cp configs/templates/reset_scene_template.json configs/reset_my_task.json
```

这个模板里最关键的字段有三类：

- `actuators`
  - 机器人和夹爪的初始控制目标。
  - 这里的名字必须和 XML 里的 actuator 名完全一致。

- `free_joints`
  - 需要显式摆放的自由物体初始位姿。
  - 常用于把暂时不用的物体移到场景下方隐藏，或者给某些对象固定初始 pose。

- `random_single_free_joint`
  - 当前最常用的一类任务初始化方式。
  - 含义是：在若干候选物体里指定一个 `active_joint`，再从 `slots` 里随机挑一个位置放过去，其余候选物体放到 `inactive_pose`。

推荐你按下面这个思路写：

1. `joints`
   - 列出这类候选物体的 free joint 名。
   - 这些名字必须都能在场景 XML 里找到。

2. `active_joint`
   - 指定默认哪一个 joint 被任务逻辑视为主对象。
   - 如果 task 代码会从 `reset_info["random_single_free_joint"]["active_joint"]` 里取对象，这个字段就必须存在。

3. `companion_joints`
   - 用来描述和主对象绑定的配套物体。
   - 比如 tube 对应 cap，或者 rack 对应 lid。
   - `pos_offset` 表示 companion 相对主对象的位置偏移。

4. `slots`
   - 定义随机摆放位置池。
   - 每个 slot 至少要有 `name`、`pos`、`quat`。

最小规范可以记成三句话：

- 名字要和 XML 一致，否则 reset 时会直接报错。
- JSON 结构要和模板一致，否则读取字段时会报错。
- 语义要和 task 逻辑一致，比如 task 默认认为 `active_joint` 是管身，那你就不要把它换成别的类型对象。

# 查看场景文件，其中reset-config是复位与初始位置随即化设置
python scripts/view_scene.py model/scenes/scene_mujoco_fast_tubes.xml   --reset-config configs/reset_single_tube_random.json

# 生成带图像的数据
python scripts/generate_tube_grasp_batch.py \
  --scene fast_tubes \
  --task tube_then_cap_grasp \
  --count 1 \
  --seed-start 40 \
  --with-images \
  --cameras overview_camera,wrist_cam,wrist_cam1 \
  --out-root data/episodes/bimanual_unscrew_video_test

# 转 mp4
python scripts/export_episode_images.py \
  data/episodes/bimanual_unscrew_video_test \
  --camera overview_camera,wrist_cam,wrist_cam1 \
  --format mp4 \
  --fps 20