"""Bimanual screw-cap task workflow.

The task module owns stage ordering, attachment lifecycle, and ScrewCapSystem
state transitions. Scene discovery, ordinary target construction, specialized
screw execution, and metadata serialization live in sibling modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...executor import TaskTargetExecutor
from ...motion_context import (
    ExecutionContext,
    ExecutionSettings,
    FixedJointState,
    GripperSettings,
    IKSettings,
    JointState,
    PlanningContext,
    SiteAttachment,
    VisualServoSettings,
    arm_motion_configs,
)
from ...mujoco_env import EnvConfig
from ...planner import TaskTargetPlanner
from ...recorder import EpisodeRecorder
from ...scene import (
    actuator_id,
    capture_free_joint_state,
    capture_site_attachment,
    set_free_joint_pose,
    site_pose,
)
from ...screw import ScrewCapSystem
from ...task import AutoLabTask, TaskConfig
from ...task_target import PlannedTaskTarget, TaskTarget
from ..common import ARM_DEFAULTS
from autolabsim.tasks.screw_cap.screw_cap_execution import (
    ScrewCapExecutionController,
    ScrewCapExecutionSettings,
)
from autolabsim.tasks.screw_cap.screw_cap_metadata import ScrewCapMetadataBuilder
from autolabsim.tasks.screw_cap.screw_cap_scene import ScrewCapSceneQuery, ScrewCapSceneState
from autolabsim.tasks.screw_cap.screw_cap_targets import ScrewCapTargetBuilder


@dataclass(frozen=True)
class BimanualUnscrewTaskConfig:
    env: EnvConfig                                  # MuJoCo 环境配置，包括模型、控制频率、渲染等基础设置。
    out_dir: Path                                   # 当前 episode 的输出目录，轨迹、状态和图像都会写到这里。
    episode_index: int                              # 当前 episode 在批量采集中的编号，用于命名和追踪样本。
    seed: int                                       # 随机种子，控制 reset 随机槽位、初始扰动等可复现因素。
    cameras: tuple[str, ...] = ("overview_camera",) # 需要记录图像的相机名称列表。
    with_images: bool = False           # 是否在采集数据时同步保存相机图像。
    tube_arm: str = "second"            # 负责横向夹住离心管本体的机械臂，默认是第二个 UR5e。
    cap_arm: str = "first"              # 负责夹住并旋拧瓶盖的机械臂，默认是第一个 UR5e。

    open_gripper: float = 0.0           # 夹爪完全打开时的控制值。
    close_gripper: float = 255.0        # 夹爪闭合夹紧时的控制值。
    cap_close_gripper: float = 150.0    # 瓶盖夹爪闭合值；不要完全闭合，否则 pad 会穿进瓶盖。
    tube_close_gripper: float = 255.0   # 管身夹爪闭合值；管身固定需要更强夹紧。

    settle_steps: int = 20       # reset 后正式动作前的静置步数，让物体和机械臂先稳定下来。
    steps_per_segment: int = 20  # 普通插值执行时，每两个路点之间拆分的仿真步数。
    grasp_hold_steps: int = 10   # 到达抓取位后、闭合夹爪前的短暂停顿步数。
    hold_steps: int = 10         # 通用保持步数，常用于动作段末尾稳定状态。
    close_steps: int = 12        # 夹爪开合动作持续的步数，调大闭合/张开会更慢。
    cap_hold_steps: int = 12     # 瓶盖放下或瓶盖相关动作完成后的保持步数。
    
    ik_max_iters: int = 500     # IK 求解最大迭代次数，调大可提高困难姿态的求解机会。
    ik_pos_tol: float = 0.0001  # IK 位置误差容忍度，越小目标点定位要求越严格。
    ik_rot_tol: float = 0.0001  # IK 姿态误差容忍度，越小末端姿态要求越严格。
    ik_damping: float = 0.001   # IK 阻尼系数，调大更稳但可能收敛更慢/精度略低。
    
    waypoint_settle_steps: int = 15  # 到达关键路点后继续保持控制的步数，用于压低动态误差。
    waypoint_settle_pos_tol: float = 0.0001  # 路点 settle 阶段允许的末端位置误差。

    visual_servo_enabled: bool = True
    visual_servo_max_iters: int = 12
    visual_servo_steps: int = 10
    visual_servo_pos_tol: float = 0.0001
    visual_servo_rot_tol: float = 0.02
    visual_servo_gain: float = 0.8
    visual_servo_integral_gain: float = 0.25
    visual_servo_max_correction: float = 0.02

    return_home_after_task: bool = True  # 任务完成后是否让两个机械臂回到 reset 后的初始姿态。
    return_home_steps: int = 40  # 两个机械臂回初始姿态的插值步数。

    release_angle: float = float(np.pi * 1.5)  # 判定瓶盖拧开的累计旋转角度阈值。
    release_lift: float = 0.008  # 旋拧过程中瓶盖脚本上升的最大高度，越小上升越慢/越少。
    thread_pitch: float = 0.008  # 模拟螺纹导程：瓶盖每转一圈沿 z 轴上升的距离。
    unscrew_steps: int = 24  # 旋拧动作段的离散步数配置，影响单段旋拧的细腻程度。
    ratchet_angle: float = float(np.pi / 2.0)  # 棘轮式旋拧每次夹住后逆时针旋转的角度。

    tube_grasp_height: float = 0.08  # 离心管本体抓取点相对管子原点的 z 方向高度。
    tube_pregrasp_distance: float = 0.10  # 管身预抓点到正式抓取点的距离，用于先靠近再夹取。
    tube_lift_offset: tuple[float, float, float] = (0.25, 0.0, 0.12)  # 管身夹住后移动/抬起的目标偏移量。
    tube_pinch_forward_offset: float = 0.0  # 管身抓取点沿夹爪前向的微调偏移。
    tube_grasp_outward_offset: float = 0.0  # 管身抓取点向外侧退让的距离，避免夹爪插得过深。
    tube_tool_roll: float = float(0)  # 管身夹爪绕自身接近方向的滚转角，影响夹爪姿态。
    cap_approach_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)  # 瓶盖夹爪接近方向，默认从正上方向下抓盖。
    cap_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 瓶盖抓取点相对瓶盖中心的世界坐标偏移。
    cap_pregrasp_distance: float = 0.02  # 瓶盖预抓点到正式抓取点的距离；过大会让当前姿态下预抓 IK 失败并产生前推后拉。
    cap_post_offset: tuple[float, float, float] = (0.0, 0.0, 0.12)  # 夹住瓶盖后先向上提起离心管/瓶盖的偏移。
    cartesian_step_size: float = 0.01  # 普通位姿阶段的末端笛卡尔插值步长，默认每 1cm 生成一个 IK 路点。
    cartesian_min_steps: int = 2  # 非零长度段最少插值点数，避免短距离阶段退化成单个关节空间跳转。
    cap_place_pos: tuple[float, float, float] = (0.18, -3.06, 0.43)  # 拧下瓶盖后放置瓶盖的世界坐标。
    cap_place_lift: float = 0.08  # 到达放盖位置前的安全抬高距离，避免平移时刮碰桌面。
    cap_clearance_lift: float = 0.05  # 瓶盖完全拧开后先向上抬起的避障高度。
    cap_tool_roll: float = float(np.pi)  # 瓶盖夹爪绕接近方向的初始滚转角，设为 pi 可避免抓盖前腕部大角度翻转。
    use_topp: bool = True  # 是否使用 TOPP 轨迹时间参数化，开启后轨迹通常更平滑。
    topp_vel: float = 1.0  # TOPP 轨迹速度约束倍率，调小会让整体运动更慢。
    topp_acc: float = 1.0  # TOPP 轨迹加速度约束倍率，调小可减少突兀加减速。


# Result 类用于将一个阶段执行完后，下一阶段需要的所有信息打包返回
@dataclass(frozen=True)
class CapLiftResult:
    plan: list[PlannedTaskTarget]
    attachments: tuple[SiteAttachment, ...]
    final_action: np.ndarray


@dataclass(frozen=True)
class TubeGraspResult:
    plan: list[PlannedTaskTarget]       # 该阶段所有规划点
    attachment: SiteAttachment          # 该阶段抓取后，tube arm 与试管之间的刚性关系
    held_tube_state: JointState         # 该阶段抓取后，试管的自由关节状态，旋拧时每一步都把试管恢复到这个状态，从而让管身保持固定。
    final_action: np.ndarray            # 阶段结束时机械臂控制值


@dataclass(frozen=True)
class UnscrewResult:
    plan: list[PlannedTaskTarget]       # 只保存旋拧计划。


@dataclass(frozen=True)
class CapPlaceResult:
    plan: list[PlannedTaskTarget]
    placed_cap_state: JointState        # 表示瓶盖放到桌面后的位置和姿态。
    release_action: np.ndarray


class BimanualUnscrewTask(AutoLabTask):
    """双机械臂旋拧开盖任务。

    主任务类只负责：
    - 编排任务阶段；
    - 创建和切换刚性 attachment；
    - 控制 ScrewCapSystem 的 engage/follow/release 生命周期；
    - 调用公共 Planner、Executor 和任务专用旋拧控制器。

    场景查询、普通 TaskTarget 构造、旋拧执行和 metadata 构造分别位于
    ``screw_cap_scene.py``、``screw_cap_targets.py``、
    ``screw_cap_execution.py`` 和 ``screw_cap_metadata.py``。
    """

    name = "bimanual_unscrew_cap"

    def __init__(self, config: BimanualUnscrewTaskConfig) -> None:
        self.runtime = config                                   # 保存配置，后续任务都通过self.runtime访问
        self.screw_system: ScrewCapSystem | None = None         # 旋拧类由于瓶盖和试管的刚性关系未确定，先初始化为 None。
        self.screw_execution: ScrewCapExecutionController | None = None

        # 调用父类，创建环境
        super().__init__(
            TaskConfig(
                env=config.env,
                with_images=config.with_images,
                cameras=config.cameras,
            )
        )

        # 公共机械臂规划与执行组件。
        self.arm_configs = arm_motion_configs(ARM_DEFAULTS)
        # IK 和 gripper 配置在 Planner 和 Executor 中共享，确保规划和执行阶段使用一致的参数。
        self.ik_settings = IKSettings(
            max_iters=config.ik_max_iters,
            pos_tol=config.ik_pos_tol,
            rot_tol=config.ik_rot_tol,
            damping=config.ik_damping,
        )
        # 夹爪值映射
        self.gripper_settings = GripperSettings(
            open_value=config.open_gripper,
            close_value=config.close_gripper,
        )
        # 创建通用规划器
        self.planner = TaskTargetPlanner(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.arm_configs,
            self.ik_settings,
            self.gripper_settings,
        )
        # 创建通用执行器，负责执行 TaskTarget、记录轨迹、计算误差等。
        self.executor = TaskTargetExecutor(
            self.env,
            self.manager, 
            self.arm_configs,
            self.ik_settings,
            self.gripper_settings,
            ExecutionSettings(
                steps_per_segment=config.steps_per_segment,
                waypoint_settle_steps=config.waypoint_settle_steps,
                waypoint_settle_pos_tol=config.waypoint_settle_pos_tol,
                visual_servo=VisualServoSettings(
                    enabled=config.visual_servo_enabled,
                    max_iters=config.visual_servo_max_iters,
                    steps=config.visual_servo_steps,
                    pos_tol=config.visual_servo_pos_tol,
                    rot_tol=config.visual_servo_rot_tol,
                    gain=config.visual_servo_gain,
                    integral_gain=config.visual_servo_integral_gain,
                    max_correction=config.visual_servo_max_correction,
                ),
            ),
        )

        # 任务专用辅助模块。
        self.scene_query = ScrewCapSceneQuery(
            self.env
        )
        self.target_builder = ScrewCapTargetBuilder(
            self.env,
            self.planner,
            ARM_DEFAULTS,
            config,
        )
        self.metadata_builder = ScrewCapMetadataBuilder(
            self.env,
            config,
            task_name=self.name,
        )

        # 与任务二保持一致，错误记录统一由公共 Executor 维护。
        self.execution_site_errors = self.executor.execution_site_errors

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """执行一次完整的双臂开盖 episode。

        流程：
        1. reset 场景并识别当前试管/瓶盖；
        2. 打开两只夹爪并记录双臂 home action；
        3. cap arm 抓住瓶盖并把整管提离试管架；
        4. tube arm 从侧面夹住管身；
        5. cap arm 通过棘轮动作旋开瓶盖；
        6. 把瓶盖放到桌面；
        7. 可选地把试管放回原槽位；
        8. 可选地让双臂回到初始姿态；
        9. 保存 episode 数组与 metadata。
        """

        self.reset()                            # 场景重置与随机化
        scene = self.scene_query.resolve()      # 识别当前试管/瓶盖的 joint/body/weld 等信息
        self._initialize_screw_system(scene)    # 创建 ScrewCapSystem，设置螺纹导程、释放角度和抬升高度

        home_action = self._settle_initial_scene()  # 初始化夹爪，记录初始位置
        recorder = EpisodeRecorder(
            self.runtime.cameras,
            self.runtime.with_images,
        )
        recorder.record(                        # 创建记录器，并记录第一帧
            self.env.get_observation(),
            np.asarray(self.env.data.ctrl, dtype=np.float64).copy(),
            "start",
        )

        # 以下是任务流程
        cap_lift = self._stage_cap_grasp_and_lift(recorder, scene)
        tube_grasp = self._stage_tube_side_grasp(
            recorder,
            scene,
            cap_lift.attachments,
        )
        unscrew = self._stage_unscrew_cap(
            recorder,
            scene,
            tube_grasp.held_tube_state,
        )
        cap_place = self._stage_place_cap(
            recorder,
            scene,
            tube_grasp.held_tube_state,
        )

        tube_return_plan = self._stage_return_tube(
            recorder,
            scene,
            tube_grasp.attachment,
            cap_place.placed_cap_state,
        )

        if self.runtime.return_home_after_task:
            self._stage_return_home(
                recorder,
                scene,
                home_action,
                cap_place.placed_cap_state,
            )

        arrays = recorder.to_arrays()
        arrays["final_state"] = self.env.get_observation()["state"]

        # 保留原任务的 metadata 语义：cap_waypoints 记录初始抓盖/抬升，
        # tube_waypoints 记录侧面夹管；放盖和归位仍保存在逐步 phase 数据中。
        metadata = self.metadata_builder.build(
            scene,
            tube_grasp.plan,
            cap_lift.plan,
            unscrew.plan,
            screw_system=self.screw_system,
            execution_site_errors=self.execution_site_errors,
            num_steps=arrays["qpos"].shape[0],
        )
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    # ------------------------------------------------------------------
    # Workflow stages
    # ------------------------------------------------------------------

    def _stage_cap_grasp_and_lift(
        self,
        recorder: EpisodeRecorder,
        scene: ScrewCapSceneState,
    ) -> CapLiftResult:
        """cap arm 抓住瓶盖，并通过瓶盖把整支试管提起。"""

        cap_pos = self.scene_query.cap_position(scene)
        # 根据试管位姿，构造稀疏目标点,pregrasp,grasp,post
        sparse = self.target_builder.cap_lift_targets(cap_pos)
        # 目标点间插入中间笛卡尔点，得到更密集的目标点
        targets = self.target_builder.densify_from_current_site(
            self.runtime.cap_arm,
            sparse.ordered(),
        )
        # IK规划
        plan = self._plan_targets(
            targets,
            default_gripper_value=self.runtime.open_gripper,
        )
        # 从IK规划结果中找到关键目标在稠密计划中的位置，并将轨迹拆分为不同的阶段
        grasp_index = self._plan_index(plan, sparse.grasp.name)
        post_index = self._plan_index(plan, sparse.post.name)
        grasp_item = plan[grasp_index]
        approach_plan = plan[: grasp_index + 1]
        lift_plan = plan[grasp_index + 1 : post_index + 1]
        # 创建固定物体上下文，cap arm 接近瓶盖时，每一步都把试管和瓶盖恢复到初始位置
        initial_context = self._execution_context(
            fixed_joint_states=(
                (
                    scene.tube_joint,
                    capture_free_joint_state(
                        self.env.model,
                        self.env.data,
                        self.env.mujoco,
                        scene.tube_joint,
                    ),
                ),
                (
                    scene.cap_joint,
                    capture_free_joint_state(
                        self.env.model,
                        self.env.data,
                        self.env.mujoco,
                        scene.cap_joint,
                    ),
                ),
            )
        )
        # 执行接近操作  要不要切换成self.executor.execute，去掉_motion_controller这一层封装？
        self._motion_controller().execute(
            recorder,
            approach_plan,
            "cap_move_to_lift_grasp",
            initial_context,
        )
        grasp_action = np.asarray(grasp_item.action, dtype=np.float64).copy()
        # 抓取动作
        self.executor.hold_action(
            recorder,
            grasp_action,
            self.runtime.grasp_hold_steps,
            "cap_settle_at_lift_grasp",
            initial_context,
        )

        self.executor.record_site_target_error(
            "cap_settle_at_lift_grasp",
            self._gripper_site(self.runtime.cap_arm),
            grasp_item.ik_site_pos,
        )

        close_cap = grasp_action.copy()
        close_cap[self._gripper_id(self.runtime.cap_arm)] = (
            self.runtime.cap_close_gripper
        )
        self.executor.move_action(
            recorder,
            close_cap,
            self.runtime.close_steps,
            "cap_close_for_lift",
            initial_context,
        )
        self.executor.record_site_target_error(
            "cap_close_for_lift",
            self._gripper_site(self.runtime.cap_arm),
            grasp_item.ik_site_pos,
        )
        # 闭合后捕获两个刚性关系：cap gripper→cap、cap gripper→tube。
        attachments = self._capture_attachments(
            self._gripper_site(self.runtime.cap_arm),
            (scene.cap_joint, scene.tube_joint),
        )
        # 记录物体与夹爪的接触关系，用于后续抬升阶段保持物体依附关系 
        lift_context = self._execution_context(attachments=attachments)

        if lift_plan:
            # 确保抬升阶段始终维持瓶盖夹爪的夹紧值。
            for item in lift_plan:
                item.action = np.asarray(item.action, dtype=np.float64).copy()
                item.action[self._gripper_id(self.runtime.cap_arm)] = (
                    self.runtime.cap_close_gripper
                )
            self._motion_controller().execute(
                recorder,
                lift_plan,
                "cap_lift_tube_out",
                lift_context,
            )
            final_action = np.asarray(
                lift_plan[-1].action,
                dtype=np.float64,
            ).copy()
        else:
            final_action = close_cap
            self.executor.move_action(
                recorder,
                final_action,
                self.runtime.steps_per_segment,
                "cap_lift_tube_out",
                lift_context,
            )
        self.executor.hold_action(
            recorder,
            final_action,
            self.runtime.hold_steps,
            "cap_hold_tube_lifted",
            lift_context,
        )
        return CapLiftResult(
            plan=plan,
            attachments=attachments,
            final_action=final_action,
        )

    def _stage_tube_side_grasp(
        self,
        recorder: EpisodeRecorder,
        scene: ScrewCapSceneState,
        cap_lift_attachments: tuple[SiteAttachment, ...],
    ) -> TubeGraspResult:
        """tube arm 在 cap arm 仍持有整管时，从侧面夹住管身。"""

        lifted_tube_pos = self.scene_query.tube_position(scene)
        sparse = self.target_builder.tube_grasp_targets(lifted_tube_pos)
        targets = self.target_builder.densify_from_current_site(
            self.runtime.tube_arm,
            sparse.approach(),
        )
        plan = self._plan_targets(
            targets,
            default_gripper_value=self.runtime.open_gripper,
        )
        grasp_index = self._plan_index(plan, sparse.grasp.name)
        grasp_item = plan[grasp_index]
        context = self._execution_context(attachments=cap_lift_attachments)

        self._motion_controller().execute(
            recorder,
            plan,
            "tube_move_to_side_grasp",
            context,
        )
        grasp_action = np.asarray(grasp_item.action, dtype=np.float64).copy()
        self.executor.hold_action(
            recorder,
            grasp_action,
            self.runtime.grasp_hold_steps,
            "tube_settle_at_side_grasp",
            context,
        )
        self.executor.record_site_target_error(
            "tube_settle_at_side_grasp",
            self._gripper_site(self.runtime.tube_arm),
            grasp_item.ik_site_pos,
        )

        close_tube = grasp_action.copy()
        close_tube[self._gripper_id(self.runtime.tube_arm)] = (
            self.runtime.tube_close_gripper
        )
        self.executor.move_action(
            recorder,
            close_tube,
            self.runtime.close_steps,
            "tube_close_side_grip",
            context,
        )
        self.executor.record_site_target_error(
            "tube_close_side_grip",
            self._gripper_site(self.runtime.tube_arm),
            grasp_item.ik_site_pos,
        )

        attachment = self._capture_attachments(
            self._gripper_site(self.runtime.tube_arm),
            (scene.tube_joint,),
        )[0]
        held_tube_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            scene.tube_joint,
        )
        return TubeGraspResult(
            plan=plan,
            attachment=attachment,
            held_tube_state=held_tube_state,
            final_action=close_tube,
        )

    def _stage_unscrew_cap(
        self,
        recorder: EpisodeRecorder,
        scene: ScrewCapSceneState,
        held_tube_state: JointState,
    ) -> UnscrewResult:
        """固定管身，驱动 cap arm 完成棘轮式旋拧。"""

        screw_system = self._require_screw_system()
        screw_system.engage(self.env)

        cap_site_pos, _ = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self._gripper_site(self.runtime.cap_arm),
        )
        start_action = np.asarray(
            self.env.data.ctrl,
            dtype=np.float64,
        ).copy()
        plan = self._motion_controller().build_unscrew_plan(
            cap_site_pos,
            start_action,
        )
        context = self._execution_context(
            fixed_joint_states=((scene.tube_joint, held_tube_state),)
        )
        self._motion_controller().execute(
            recorder,
            plan,
            "cap_unscrew",
            context,
            allow_topp=False,
        )

        # 瓶盖达到释放角后，后续搬运阶段让瓶盖跟随 cap gripper。
        screw_system.start_follow_after_release(self.env)
        return UnscrewResult(plan=plan)

    def _stage_place_cap(
        self,
        recorder: EpisodeRecorder,
        scene: ScrewCapSceneState,
        held_tube_state: JointState,
    ) -> CapPlaceResult:
        """把已经旋开的瓶盖移到桌面并释放。"""

        cap_site_pos, cap_site_quat = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self._gripper_site(self.runtime.cap_arm),
        )

        sparse = self.target_builder.cap_place_targets(
            cap_site_pos,
            cap_site_quat,
        )
        targets = self.target_builder.densify_from_current_site(
            self.runtime.cap_arm,
            sparse.ordered(),
        )
        plan = self._plan_targets(
            targets,
            default_gripper_value=self.runtime.cap_close_gripper,
        )
        tube_context = self._execution_context(
            fixed_joint_states=((scene.tube_joint, held_tube_state),)
        )
        self._motion_controller().execute(
            recorder,
            plan,
            "cap_place_on_table",
            tube_context,
        )

        screw_system = self._require_screw_system()
        screw_system.release_follow()
        placed_cap_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            scene.cap_joint,
        )

        open_cap = np.asarray(plan[-1].action, dtype=np.float64).copy()
        open_cap[self._gripper_id(self.runtime.cap_arm)] = (
            self.runtime.open_gripper
        )
        release_context = self._execution_context(
            fixed_joint_states=(
                (scene.tube_joint, held_tube_state),
                (scene.cap_joint, placed_cap_state),
            )
        )
        self.executor.move_action(
            recorder,
            open_cap,
            self.runtime.close_steps,
            "cap_release_on_table",
            release_context,
        )
        return CapPlaceResult(
            plan=plan,
            placed_cap_state=placed_cap_state,
            release_action=open_cap,
        )

    def _stage_return_tube(
        self,
        recorder: EpisodeRecorder,
        scene: ScrewCapSceneState,
        tube_attachment: SiteAttachment,
        placed_cap_state: JointState,
    ) -> list[PlannedTaskTarget]:
        """让 tube arm 把试管放回 reset 时选中的原槽位。"""

        sparse = self.target_builder.tube_return_targets(scene.slot_pos)
        targets = self.target_builder.densify_from_current_site(
            self.runtime.tube_arm,
            sparse.ordered(),
        )
        plan = self._plan_targets(
            targets,
            default_gripper_value=self.runtime.tube_close_gripper,
        )
        return_context = self._execution_context(
            fixed_joint_states=((scene.cap_joint, placed_cap_state),),
            attachments=(tube_attachment,),
        )
        self._motion_controller().execute(
            recorder,
            plan,
            "tube_return_to_rack",
            return_context,
        )

        if scene.slot_quat is not None:
            # 保留原行为：到达放置目标后把 free joint 精确对齐到 reset 槽位。
            set_free_joint_pose(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                scene.tube_joint,
                scene.slot_pos,
                scene.slot_quat,
            )

        open_tube = np.asarray(plan[-1].action, dtype=np.float64).copy()
        open_tube[self._gripper_id(self.runtime.tube_arm)] = (
            self.runtime.open_gripper
        )
        cap_fixed_context = self._execution_context(
            fixed_joint_states=((scene.cap_joint, placed_cap_state),)
        )
        self.executor.move_action(
            recorder,
            open_tube,
            self.runtime.close_steps,
            "tube_release_in_rack",
            cap_fixed_context,
        )
        self.executor.hold_action(
            recorder,
            open_tube,
            self.runtime.hold_steps,
            "tube_hold_released",
            cap_fixed_context,
        )
        return plan

    def _stage_return_home(
        self,
        recorder: EpisodeRecorder,
        scene: ScrewCapSceneState,
        home_action: np.ndarray,
        placed_cap_state: JointState,
    ) -> None:
        """固定最终物体状态，并把两只机械臂返回 reset 后姿态。"""

        final_tube_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            scene.tube_joint,
        )
        context = self._execution_context(
            fixed_joint_states=(
                (scene.cap_joint, placed_cap_state),
                (scene.tube_joint, final_tube_state),
            )
        )
        self.executor.move_action(
            recorder,
            np.asarray(home_action, dtype=np.float64),
            self.runtime.return_home_steps,
            "both_arms_return_home",
            context,
        )

    # ------------------------------------------------------------------
    # Initialization and shared helpers
    # ------------------------------------------------------------------

    def _initialize_screw_system(self, scene: ScrewCapSceneState) -> None:
        """为当前随机选中的 tube-cap 对创建 ScrewCapSystem。"""

        self.screw_system = ScrewCapSystem(
            tube_joint=scene.tube_joint,
            cap_joint=scene.cap_joint,
            cap_site=self._gripper_site(self.runtime.cap_arm),
            weld_name=scene.cap_weld,
            release_angle=self.runtime.release_angle,
            thread_pitch=self.runtime.thread_pitch,
            max_lift=self.runtime.release_lift,
        )
        self.manager.systems = [self.screw_system]
        # 系统是在 env.reset() 之后根据随机对象创建的，因此需要显式 on_reset。
        self.screw_system.on_reset(self.env)

        self.screw_execution = ScrewCapExecutionController(
            self.env,
            self.manager,
            self.planner,
            self.executor,
            self.target_builder,
            self.screw_system,
            ARM_DEFAULTS,
            ScrewCapExecutionSettings.from_runtime(self.runtime),
        )
    
    def _settle_initial_scene(self) -> np.ndarray:
        """打开两只夹爪、等待场景稳定，并返回双臂 home action。"""

        action = np.asarray(
            self.env.data.ctrl,
            dtype=np.float64,
        ).copy()

        action[self._gripper_id(self.runtime.tube_arm)] = (
            self.runtime.open_gripper
        )
        action[self._gripper_id(self.runtime.cap_arm)] = (
            self.runtime.open_gripper
        )

        for _ in range(max(0, int(self.runtime.settle_steps))):
            self.manager.step(action)

        return np.asarray(
            self.env.data.ctrl,
            dtype=np.float64,
        ).copy()

    def _plan_targets(
        self,
        targets: list[TaskTarget] | tuple[TaskTarget, ...],
        *,
        default_gripper_value: float,
        planning_context: PlanningContext | None = None,
    ) -> list[PlannedTaskTarget]:
        return self.planner.plan(
            targets,
            planning_context or PlanningContext(),
            default_gripper_value=default_gripper_value,
        )

    @staticmethod
    def _plan_index(
        plan: list[PlannedTaskTarget],
        waypoint_name: str,
    ) -> int:
        for index, item in enumerate(plan):
            if item.name == waypoint_name:
                return index
        raise ValueError(f"Planned waypoint not found: {waypoint_name}")

    def _capture_attachments(
        self,
        parent_site: str,
        joint_names: tuple[str, ...],
    ) -> tuple[SiteAttachment, ...]:
        return tuple(
            SiteAttachment.from_mapping(
                joint_name,
                parent_site,
                capture_site_attachment(
                    self.env.model,
                    self.env.data,
                    self.env.mujoco,
                    joint_name,
                    parent_site,
                ),
            )
            for joint_name in joint_names
        )

    @staticmethod
    def _execution_context(
        *,
        fixed_joint_states: tuple[tuple[str, JointState], ...] = (),
        attachments: tuple[SiteAttachment, ...] = (),
    ) -> ExecutionContext:
        return ExecutionContext(
            fixed_joint_states=tuple(
                FixedJointState(joint_name=joint_name, state=state)
                for joint_name, state in fixed_joint_states
            ),
            attachments=attachments,
        )

    def _motion_controller(self) -> ScrewCapExecutionController:
        if self.screw_execution is None:
            raise RuntimeError("Screw execution controller is not initialized")
        return self.screw_execution

    def _require_screw_system(self) -> ScrewCapSystem:
        if self.screw_system is None:
            raise RuntimeError("ScrewCapSystem is not initialized")
        return self.screw_system

    def _gripper_site(self, arm_name: str) -> str:
        return str(ARM_DEFAULTS[arm_name]["gripper_site"])

    def _gripper_id(self, arm_name: str) -> int:
        return actuator_id(
            self.env.model,
            self.env.mujoco,
            str(ARM_DEFAULTS[arm_name]["gripper_actuator"]),
        )
