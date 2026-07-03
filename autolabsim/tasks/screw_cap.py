'''
双臂旋拧开盖任务
'''
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..ik import solve_site_ik
from ..math3d import gripper_quat_from_axes, normalize_quat, unit
from ..mujoco_env import EnvConfig
from ..recorder import EpisodeRecorder
from ..scene_profile import active_joint_fallback
from ..scene import (
    apply_site_attachment,
    actuator_id,
    body_pos,
    capture_free_joint_state,
    capture_site_attachment,
    free_joint_pos,
    joint_qpos_ids,
    restore_free_joint_state,
    set_free_joint_pose,
    site_pose,
)
from ..screw import ScrewCapSystem
from ..task import AutoLabTask, TaskConfig
from ..topp import Topp, ToppConfig
from .common import (
    ARM_DEFAULTS,
    cap_body_from_tube_joint,
    cap_joint_from_tube_joint,
    cap_weld_from_tube_joint,
    json_safe,
    random_reset_info,
)


@dataclass(frozen=True)
class BimanualUnscrewTaskConfig:
    env: EnvConfig  # MuJoCo 环境配置，包括模型、控制频率、渲染等基础设置。
    out_dir: Path  # 当前 episode 的输出目录，轨迹、状态和图像都会写到这里。
    episode_index: int  # 当前 episode 在批量采集中的编号，用于命名和追踪样本。
    seed: int  # 随机种子，控制 reset 随机槽位、初始扰动等可复现因素。
    cameras: tuple[str, ...] = ("overview_camera",)  # 需要记录图像的相机名称列表。
    with_images: bool = False  # 是否在采集数据时同步保存相机图像。
    tube_arm: str = "second"  # 负责横向夹住离心管本体的机械臂，默认是第二个 UR5e。
    cap_arm: str = "first"  # 负责夹住并旋拧瓶盖的机械臂，默认是第一个 UR5e。
    open_gripper: float = 0.0  # 夹爪完全打开时的控制值。
    close_gripper: float = 255.0  # 夹爪闭合夹紧时的控制值。
    cap_close_gripper: float = 150.0  # 瓶盖夹爪闭合值；不要完全闭合，否则 pad 会穿进瓶盖。
    tube_close_gripper: float = 255.0  # 管身夹爪闭合值；管身固定需要更强夹紧。
    settle_steps: int = 20  # reset 后正式动作前的静置步数，让物体和机械臂先稳定下来。
    steps_per_segment: int = 20  # 普通插值执行时，每两个路点之间拆分的仿真步数。
    grasp_hold_steps: int = 10  # 到达抓取位后、闭合夹爪前的短暂停顿步数。
    hold_steps: int = 10  # 通用保持步数，常用于动作段末尾稳定状态。
    close_steps: int = 12  # 夹爪开合动作持续的步数，调大闭合/张开会更慢。
    cap_hold_steps: int = 12  # 瓶盖放下或瓶盖相关动作完成后的保持步数。
    return_home_after_task: bool = True  # 任务完成后是否让两个机械臂回到 reset 后的初始姿态。
    return_home_steps: int = 40  # 两个机械臂回初始姿态的插值步数。
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
    release_angle: float = float(np.pi * 1.5)  # 判定瓶盖拧开的累计旋转角度阈值。
    release_lift: float = 0.008  # 旋拧过程中瓶盖脚本上升的最大高度，越小上升越慢/越少。
    thread_pitch: float = 0.008  # 模拟螺纹导程：瓶盖每转一圈沿 z 轴上升的距离。
    unscrew_steps: int = 24  # 旋拧动作段的离散步数配置，影响单段旋拧的细腻程度。
    ratchet_angle: float = float(np.pi / 2.0)  # 棘轮式旋拧每次夹住后逆时针旋转的角度。
    return_tube_to_rack: bool = True  # 完成开盖后，是否把离心管放回试管架。
    ik_max_iters: int = 500  # IK 求解最大迭代次数，调大可提高困难姿态的求解机会。
    ik_pos_tol: float = 0.0001  # IK 位置误差容忍度，越小目标点定位要求越严格。
    ik_rot_tol: float = 0.0001  # IK 姿态误差容忍度，越小末端姿态要求越严格。
    ik_damping: float = 0.001   # IK 阻尼系数，调大更稳但可能收敛更慢/精度略低。
    waypoint_settle_steps: int = 15  # 到达关键路点后继续保持控制的步数，用于压低动态误差。
    waypoint_settle_pos_tol: float = 0.0001  # 路点 settle 阶段允许的末端位置误差。
    use_topp: bool = True  # 是否使用 TOPP 轨迹时间参数化，开启后轨迹通常更平滑。
    topp_vel: float = 1.0  # TOPP 轨迹速度约束倍率，调小会让整体运动更慢。
    topp_acc: float = 1.0  # TOPP 轨迹加速度约束倍率，调小可减少突兀加减速。


class BimanualUnscrewTask(AutoLabTask):
    name = "bimanual_unscrew_cap"

    def __init__(self, config: BimanualUnscrewTaskConfig):
        self.runtime = config
        self.tube_arm = ARM_DEFAULTS[config.tube_arm]
        self.cap_arm = ARM_DEFAULTS[config.cap_arm]
        self.screw_system: ScrewCapSystem | None = None
        self.execution_site_errors: list[dict[str, Any]] = []
        super().__init__(
            TaskConfig(
                env=config.env,
                with_images=config.with_images,
                cameras=config.cameras,
            )
        )

    def run(self) -> dict[str, Any]:
        obs = self.reset()
        reset_info = dict(self.env.last_reset_info)
        random_info = random_reset_info(reset_info)
        active_joint = str(random_info["active_joint"]) if random_info else active_joint_fallback()
        active_cap_joint = cap_joint_from_tube_joint(active_joint)
        active_cap_body = cap_body_from_tube_joint(active_joint)
        active_cap_weld = cap_weld_from_tube_joint(active_joint)
        self.screw_system = ScrewCapSystem(
            tube_joint=active_joint,
            cap_joint=active_cap_joint,
            cap_site=str(self.cap_arm["gripper_site"]),
            weld_name=active_cap_weld,
            release_angle=self.runtime.release_angle,
            thread_pitch=self.runtime.thread_pitch,
            max_lift=self.runtime.release_lift,
        )
        self.manager.systems = [self.screw_system]
        self.screw_system.on_reset(self.env)

        tube_gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.tube_arm["gripper_actuator"]))
        cap_gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.cap_arm["gripper_actuator"]))

        settle_action = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        settle_action[tube_gripper_id] = self.runtime.open_gripper
        settle_action[cap_gripper_id] = self.runtime.open_gripper
        for _ in range(self.runtime.settle_steps):
            obs, *_ = self.manager.step(settle_action)
        home_action = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()

        cap_pos = body_pos(self.env.model, self.env.data, self.env.mujoco, active_cap_body)
        recorder = EpisodeRecorder(self.runtime.cameras, self.runtime.with_images)

        cap_sparse_waypoints = self._make_cap_waypoints(cap_pos)
        cap_waypoints = self._densify_waypoints_from_current_site(self.cap_arm, cap_sparse_waypoints)
        cap_plan = self._plan_arm(self.cap_arm, cap_waypoints, self.runtime.open_gripper)
        cap_grasp_index = self._plan_index(cap_plan, "cap_grasp")
        cap_post_index = self._plan_index(cap_plan, "cap_post")
        cap_grasp_item = cap_plan[cap_grasp_index]
        cap_lift_plan = cap_plan[cap_grasp_index + 1 : cap_post_index + 1]

        recorder.record(self.env.get_observation(), np.asarray(self.env.data.ctrl).copy(), "start")

        initial_tube_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_joint)
        initial_cap_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_cap_joint)
        initial_object_states = [(active_joint, initial_tube_state), (active_cap_joint, initial_cap_state)]
        self._execute_plan(
            recorder,
            cap_plan[: cap_grasp_index + 1],
            "cap_move_to_lift_grasp",
            fixed_joint_states=initial_object_states,
            debug_site=str(self.cap_arm["gripper_site"]),
        )
        cap_grasp_action = np.asarray(cap_grasp_item["action"]).copy()
        self._hold_action(
            recorder,
            cap_grasp_action,
            self.runtime.grasp_hold_steps,
            "cap_settle_at_lift_grasp",
            fixed_joint_states=initial_object_states,
        )
        self._record_site_target_error("cap_settle_at_lift_grasp", str(self.cap_arm["gripper_site"]), cap_grasp_item["target_pos"])
        close_cap = np.asarray(cap_grasp_item["action"]).copy()
        close_cap[cap_gripper_id] = self.runtime.cap_close_gripper
        self._move_action(
            recorder,
            close_cap,
            self.runtime.close_steps,
            "cap_close_for_lift",
            fixed_joint_states=initial_object_states,
        )
        self._record_site_target_error("cap_close_for_lift", str(self.cap_arm["gripper_site"]), cap_grasp_item["target_pos"])

        cap_lift_attachments = self._attachments_to_site(
            str(self.cap_arm["gripper_site"]),
            (active_cap_joint, active_joint),
        )
        cap_lift_action = np.asarray((cap_lift_plan[-1] if cap_lift_plan else cap_grasp_item)["action"]).copy()
        cap_lift_action[cap_gripper_id] = self.runtime.cap_close_gripper
        if cap_lift_plan:
            for item in cap_lift_plan:
                item["action"] = np.asarray(item["action"]).copy()
                item["action"][cap_gripper_id] = self.runtime.cap_close_gripper
            self._execute_plan(
                recorder,
                cap_lift_plan,
                "cap_lift_tube_out",
                follow_attachments=cap_lift_attachments,
                debug_site=str(self.cap_arm["gripper_site"]),
            )
        else:
            self._move_action(
                recorder,
                cap_lift_action,
                self.runtime.steps_per_segment,
                "cap_lift_tube_out",
                follow_attachments=cap_lift_attachments,
            )
        self._hold_action(
            recorder,
            cap_lift_action,
            self.runtime.hold_steps,
            "cap_hold_tube_lifted",
            follow_attachments=cap_lift_attachments,
        )

        lifted_tube_pos = free_joint_pos(self.env.model, self.env.data, self.env.mujoco, active_joint)
        tube_waypoints = self._densify_waypoints_from_current_site(
            self.tube_arm,
            self._make_tube_waypoints(lifted_tube_pos)[:2],
        )
        tube_plan = self._plan_arm(self.tube_arm, tube_waypoints, self.runtime.open_gripper)
        tube_grasp_index = self._plan_index(tube_plan, "tube_grasp")
        tube_grasp_item = tube_plan[tube_grasp_index]

        self._execute_plan(
            recorder,
            tube_plan,
            "tube_move_to_side_grasp",
            follow_attachments=cap_lift_attachments,
            debug_site=str(self.tube_arm["gripper_site"]),
        )
        tube_grasp_action = np.asarray(tube_grasp_item["action"]).copy()
        self._hold_action(
            recorder,
            tube_grasp_action,
            self.runtime.grasp_hold_steps,
            "tube_settle_at_side_grasp",
            follow_attachments=cap_lift_attachments,
        )
        self._record_site_target_error("tube_settle_at_side_grasp", str(self.tube_arm["gripper_site"]), tube_grasp_item["target_pos"])
        close_tube = np.asarray(tube_grasp_item["action"]).copy()
        close_tube[tube_gripper_id] = self.runtime.tube_close_gripper
        self._move_action(
            recorder,
            close_tube,
            self.runtime.close_steps,
            "tube_close_side_grip",
            follow_attachments=cap_lift_attachments,
        )
        self._record_site_target_error("tube_close_side_grip", str(self.tube_arm["gripper_site"]), tube_grasp_item["target_pos"])

        tube_grip_attachment = self._attachments_to_site(str(self.tube_arm["gripper_site"]), (active_joint,))
        held_tube_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_joint)

        self.screw_system.engage(self.env)
        cap_site_pos, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, str(self.cap_arm["gripper_site"]))
        unscrew_start_action = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        unscrew_plan = self._plan_unscrew({"pos": cap_site_pos}, unscrew_start_action)
        self._execute_plan(recorder, unscrew_plan, "cap_unscrew", hold_joint=active_joint, held_state=held_tube_state)
        if self.screw_system is not None:
            self.screw_system.start_follow_after_release(self.env)

        cap_site_pos_after_unscrew, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, str(self.cap_arm["gripper_site"]))
        cap_place_waypoints = self._densify_waypoints_from_current_site(
            self.cap_arm,
            self._make_cap_place_waypoints(cap_site_pos_after_unscrew),
        )
        cap_place_plan = self._plan_arm(self.cap_arm, cap_place_waypoints, self.runtime.cap_close_gripper)
        self._execute_plan(
            recorder,
            cap_place_plan,
            "cap_place_on_table",
            hold_joint=active_joint,
            held_state=held_tube_state,
            debug_site=str(self.cap_arm["gripper_site"]),
        )
        if self.screw_system is not None:
            self.screw_system.release_follow()
        placed_cap_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_cap_joint)
        open_cap = np.asarray(cap_place_plan[-1]["action"]).copy()
        open_cap[cap_gripper_id] = self.runtime.open_gripper
        self._move_action(
            recorder,
            open_cap,
            self.runtime.close_steps,
            "cap_release_on_table",
            hold_joint=active_joint,
            held_state=held_tube_state,
            fixed_joint_states=[(active_cap_joint, placed_cap_state)],
        )

        if self.runtime.return_tube_to_rack:
            slot_pose = random_info.get("pose") if random_info else None
            slot_pos = np.asarray(slot_pose["pos"], dtype=np.float64) if isinstance(slot_pose, dict) else free_joint_pos(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                active_joint,
            )
            slot_quat = np.asarray(slot_pose.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64) if isinstance(slot_pose, dict) else None
            return_waypoints = self._densify_waypoints_from_current_site(
                self.tube_arm,
                self._make_tube_return_waypoints(slot_pos),
            )
            return_plan = self._plan_arm(self.tube_arm, return_waypoints, self.runtime.tube_close_gripper)
            self._execute_plan(
                recorder,
                return_plan,
                "tube_return_to_rack",
                follow_attachments=tube_grip_attachment,
                fixed_joint_states=[(active_cap_joint, placed_cap_state)],
                debug_site=str(self.tube_arm["gripper_site"]),
            )
            if slot_quat is not None:
                set_free_joint_pose(self.env.model, self.env.data, self.env.mujoco, active_joint, slot_pos, slot_quat)
            open_tube = np.asarray(return_plan[-1]["action"]).copy()
            open_tube[tube_gripper_id] = self.runtime.open_gripper
            self._move_action(
                recorder,
                open_tube,
                self.runtime.close_steps,
                "tube_release_in_rack",
                fixed_joint_states=[(active_cap_joint, placed_cap_state)],
            )
            self._hold_action(
                recorder,
                open_tube,
                self.runtime.hold_steps,
                "tube_hold_released",
                fixed_joint_states=[(active_cap_joint, placed_cap_state)],
            )

        if self.runtime.return_home_after_task:
            final_fixed_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = [(active_cap_joint, placed_cap_state)]
            final_fixed_states.append(
                (
                    active_joint,
                    capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_joint),
                )
            )
            self._move_action(
                recorder,
                home_action,
                self.runtime.return_home_steps,
                "both_arms_return_home",
                fixed_joint_states=final_fixed_states,
            )

        arrays = recorder.to_arrays()
        arrays["final_state"] = self.env.get_observation()["state"]
        metadata = self._make_metadata(
            reset_info,
            random_info,
            active_joint,
            active_cap_joint,
            active_cap_body,
            tube_plan,
            cap_plan,
            unscrew_plan,
            num_steps=arrays["qpos"].shape[0],
        )
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    def _make_tube_waypoints(self, tube_pos: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(self.tube_arm["approach_axis"], "tube_approach_axis")
        closing = unit(self.tube_arm["closing_axis"], "tube_closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.tube_tool_roll)
        grasp_pos = (
            tube_pos
            + np.asarray([0.0, 0.0, self.runtime.tube_grasp_height], dtype=np.float64)
            + approach * self.runtime.tube_pinch_forward_offset
            - approach * self.runtime.tube_grasp_outward_offset
        )
        pregrasp_pos = grasp_pos - approach * self.runtime.tube_pregrasp_distance
        lift_pos = grasp_pos + np.asarray(self.runtime.tube_lift_offset, dtype=np.float64)
        return [
            {"name": "tube_pregrasp", "pos": pregrasp_pos, "quat": quat},
            {"name": "tube_grasp", "pos": grasp_pos, "quat": quat},
            {"name": "tube_lift", "pos": lift_pos, "quat": quat},
        ]

    def _make_cap_waypoints(self, cap_pos: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(np.asarray(self.runtime.cap_approach_axis, dtype=np.float64), "cap_approach_axis")
        closing = unit(self.cap_arm["closing_axis"], "cap_closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.cap_tool_roll)
        grasp_pos = cap_pos + np.asarray(self.runtime.cap_offset, dtype=np.float64)
        pregrasp_pos = grasp_pos - approach * self.runtime.cap_pregrasp_distance
        post_pos = grasp_pos + np.asarray(self.runtime.cap_post_offset, dtype=np.float64)
        return [
            {"name": "cap_pregrasp", "pos": pregrasp_pos, "quat": quat},
            {"name": "cap_grasp", "pos": grasp_pos, "quat": quat},
            {"name": "cap_post", "pos": post_pos, "quat": quat},
        ]

    def _make_cap_place_waypoints(self, current_cap_grasp_pos: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(np.asarray(self.runtime.cap_approach_axis, dtype=np.float64), "cap_place_approach_axis")
        closing = unit(self.cap_arm["closing_axis"], "cap_place_closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.cap_tool_roll)
        place_pos = np.asarray(self.runtime.cap_place_pos, dtype=np.float64)
        preplace_pos = place_pos + np.asarray([0.0, 0.0, self.runtime.cap_place_lift], dtype=np.float64)
        clear_pos = np.asarray(current_cap_grasp_pos, dtype=np.float64) + np.asarray([0.0, 0.0, self.runtime.cap_clearance_lift], dtype=np.float64)
        return [
            {"name": "cap_clearance_lift", "pos": clear_pos, "quat": quat},
            {"name": "cap_preplace", "pos": preplace_pos, "quat": quat},
            {"name": "cap_place", "pos": place_pos, "quat": quat},
        ]

    def _make_tube_return_waypoints(self, slot_pos: np.ndarray) -> list[dict[str, Any]]:
        return_pos = self._make_tube_waypoints(np.asarray(slot_pos, dtype=np.float64))[1]["pos"]
        _, current_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, str(self.tube_arm["gripper_site"]))
        preplace_pos = return_pos + np.asarray([0.0, 0.0, 0.10], dtype=np.float64)
        return [
            {"name": "tube_return_preplace", "pos": preplace_pos, "quat": current_quat},
            {"name": "tube_return_place", "pos": return_pos, "quat": current_quat},
        ]

    def _attachments_to_site(self, site_name: str, joint_names: tuple[str, ...]) -> list[tuple[str, str, dict[str, np.ndarray]]]:
        return [
            (
                joint_name,
                site_name,
                capture_site_attachment(self.env.model, self.env.data, self.env.mujoco, joint_name, site_name),
            )
            for joint_name in joint_names
        ]

    def _apply_follow_attachments(self, attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None) -> None:
        if not attachments:
            return
        for joint_name, site_name, attachment in attachments:
            apply_site_attachment(self.env.model, self.env.data, self.env.mujoco, joint_name, site_name, attachment)

    def _restore_fixed_joint_states(self, states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None) -> None:
        if not states:
            return
        for joint_name, state in states:
            restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, joint_name, state)

    def _refresh_screw_pose_after_constraints(self, action: np.ndarray, obs: dict[str, Any]) -> dict[str, Any]:
        if self.screw_system is None or not self.screw_system.progress.engaged or self.screw_system.progress.released:
            return obs
        self.screw_system.after_step(self.env, action, obs)
        return self.env.get_observation()

    @staticmethod
    def _quat_slerp(start_quat: np.ndarray, target_quat: np.ndarray, alpha: float) -> np.ndarray:
        start = normalize_quat(start_quat)
        target = normalize_quat(target_quat)
        dot = float(np.dot(start, target))
        if dot < 0.0:
            target = -target
            dot = -dot
        if dot > 0.9995:
            return normalize_quat((1.0 - alpha) * start + alpha * target)
        theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta = float(np.sin(theta))
        lhs = np.sin((1.0 - alpha) * theta) / sin_theta
        rhs = np.sin(alpha * theta) / sin_theta
        return normalize_quat(lhs * start + rhs * target)

    def _densify_cartesian_waypoints(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        waypoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        dense: list[dict[str, Any]] = []
        current_pos = np.asarray(start_pos, dtype=np.float64)
        current_quat = normalize_quat(start_quat)
        for index, waypoint in enumerate(waypoints):
            target_pos = np.asarray(waypoint["pos"], dtype=np.float64)
            target_quat = normalize_quat(waypoint["quat"])
            distance = float(np.linalg.norm(target_pos - current_pos))
            quat_delta = 1.0 - abs(float(np.dot(current_quat, target_quat)))
            if distance < 1e-8 and quat_delta < 1e-8:
                steps = 1
            else:
                steps = max(
                    int(self.runtime.cartesian_min_steps),
                    int(np.ceil(distance / max(1e-6, float(self.runtime.cartesian_step_size)))),
                )
            for step in range(1, steps + 1):
                alpha = step / steps
                name = str(waypoint["name"]) if step == steps else f"{waypoint['name']}_cart_{step:02d}"
                dense.append(
                    {
                        "name": name,
                        "pos": (1.0 - alpha) * current_pos + alpha * target_pos,
                        "quat": self._quat_slerp(current_quat, target_quat, alpha),
                    }
                )
            current_pos = target_pos
            current_quat = target_quat
        return dense

    def _densify_waypoints_from_current_site(
        self,
        arm: dict[str, Any],
        waypoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        start_pos, start_quat = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            str(arm["gripper_site"]),
        )
        return self._densify_cartesian_waypoints(start_pos, start_quat, waypoints)

    @staticmethod
    def _plan_index(plan: list[dict[str, Any]], waypoint_name: str) -> int:
        for index, item in enumerate(plan):
            if item.get("name") == waypoint_name:
                return index
        raise ValueError(f"Planned waypoint not found: {waypoint_name}")

    def _plan_arm(self, arm: dict[str, Any], waypoints: list[dict[str, Any]], gripper_value: float) -> list[dict[str, Any]]:
        joint_names = tuple(arm["joint_names"])
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        arm_actuator_ids = [actuator_id(self.env.model, self.env.mujoco, name) for name in joint_names]
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(arm["gripper_actuator"]))

        start_qpos = self.env.data.qpos.copy()
        start_qvel = self.env.data.qvel.copy()
        start_ctrl = self.env.data.ctrl.copy()
        plan: list[dict[str, Any]] = []
        for waypoint in waypoints:
            result = solve_site_ik(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                str(arm["gripper_site"]),
                joint_names,
                waypoint["pos"],
                waypoint["quat"],
                max_iters=self.runtime.ik_max_iters,
                pos_tol=self.runtime.ik_pos_tol,
                rot_tol=self.runtime.ik_rot_tol,
                damping=self.runtime.ik_damping,
            )
            action = start_ctrl.copy()
            for action_id, qpos_id in zip(arm_actuator_ids, qpos_ids, strict=True):
                action[action_id] = result.qpos[qpos_id]
            action[gripper_id] = gripper_value
            plan.append(
                {
                    "name": waypoint["name"],
                    "target_pos": waypoint["pos"].tolist(),
                    "target_quat_wxyz": waypoint["quat"].tolist(),
                    "action": action,
                    "ik_success": result.success,
                    "ik_pos_error": result.pos_error,
                    "ik_rot_error": result.rot_error,
                    "arm_joint_names": list(joint_names),
                    "arm_qpos": result.qpos[qpos_ids].tolist(),
                }
            )
        self.env.data.qpos[:] = start_qpos
        self.env.data.qvel[:] = start_qvel
        self.env.data.ctrl[:] = start_ctrl
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
        return plan

    def _plan_unscrew(self, grasp_waypoint: dict[str, Any], start_action: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(np.asarray(self.runtime.cap_approach_axis, dtype=np.float64), "cap_unscrew_axis")
        closing = unit(self.cap_arm["closing_axis"], "cap_closing_axis")
        pos = np.asarray(grasp_waypoint["pos"], dtype=np.float64)
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.cap_arm["gripper_actuator"]))
        ratchet_angle = max(1e-6, float(self.runtime.ratchet_angle))
        loops = max(1, int(np.ceil(self.runtime.release_angle / ratchet_angle)))
        plan: list[dict[str, Any]] = []
        current_action = np.asarray(start_action, dtype=np.float64).copy()
        accumulated_twist = 0.0

        def pos_for_twist(angle: float) -> np.ndarray:
            lifted_pos = pos.copy()
            lifted_pos[2] += min(
                float(self.runtime.release_lift),
                float(self.runtime.thread_pitch) * (float(angle) / (2.0 * np.pi)),
            )
            return lifted_pos

        for loop_id in range(1, loops + 1):
            segment_angle = min(ratchet_angle, self.runtime.release_angle - accumulated_twist)
            twist_angle = accumulated_twist + segment_angle
            target_pos = pos_for_twist(twist_angle)
            rotate_quat = gripper_quat_from_axes(
                self.env.mujoco,
                approach,
                closing,
                self.runtime.cap_tool_roll - segment_angle,
            )
            rotate_plan = self._plan_arm(
                self.cap_arm,
                [
                    {
                        "name": f"cap_ratchet_twist_{loop_id:02d}",
                        "pos": target_pos,
                        "quat": normalize_quat(rotate_quat),
                    }
                ],
                self.runtime.cap_close_gripper,
            )[0]
            rotate_plan["twist_angle"] = float(twist_angle)
            plan.append(rotate_plan)
            current_action = np.asarray(rotate_plan["action"], dtype=np.float64).copy()
            accumulated_twist = twist_angle

            if twist_angle >= self.runtime.release_angle - 1e-9:
                continue

            open_action = current_action.copy()
            open_action[gripper_id] = self.runtime.open_gripper
            plan.append(
                {
                    "name": f"cap_ratchet_open_{loop_id:02d}",
                    "target_pos": target_pos.tolist(),
                    "target_quat_wxyz": rotate_quat.tolist(),
                    "action": open_action,
                    "ik_success": True,
                    "ik_pos_error": 0.0,
                    "ik_rot_error": 0.0,
                    "twist_angle": float(twist_angle),
                    "steps": self.runtime.close_steps,
                }
            )

            rewind_quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.cap_tool_roll)
            rewind_plan = self._plan_arm(
                self.cap_arm,
                [
                    {
                        "name": f"cap_ratchet_rewind_{loop_id:02d}",
                        "pos": target_pos,
                        "quat": normalize_quat(rewind_quat),
                    }
                ],
                self.runtime.open_gripper,
            )[0]
            rewind_plan["twist_angle"] = float(twist_angle)
            plan.append(rewind_plan)

            close_action = np.asarray(rewind_plan["action"], dtype=np.float64).copy()
            close_action[gripper_id] = self.runtime.cap_close_gripper
            plan.append(
                {
                    "name": f"cap_ratchet_regrip_{loop_id:02d}",
                    "target_pos": target_pos.tolist(),
                    "target_quat_wxyz": rewind_quat.tolist(),
                    "action": close_action,
                    "ik_success": True,
                    "ik_pos_error": 0.0,
                    "ik_rot_error": 0.0,
                    "twist_angle": float(twist_angle),
                    "steps": self.runtime.close_steps,
                }
            )
            current_action = close_action
        return plan

    def _execute_plan(
        self,
        recorder: EpisodeRecorder,
        plan: list[dict[str, Any]],
        phase_prefix: str,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        debug_site: str | None = None,
    ) -> None:
        can_use_topp = (
            self.runtime.use_topp
            and len(plan) >= 2
            and all("arm_joint_names" in item and "arm_qpos" in item for item in plan)
            and all("steps" not in item for item in plan)
        )
        if can_use_topp:
            self._execute_topp_plan(
                recorder,
                plan,
                phase_prefix,
                hold_joint=hold_joint,
                held_state=held_state,
                follow_attachments=follow_attachments,
                fixed_joint_states=fixed_joint_states,
                debug_site=debug_site,
            )
            return

        for item in plan:
            phase = f"{phase_prefix}:{item['name']}"
            self._move_action(
                recorder,
                np.asarray(item["action"]),
                int(item.get("steps", self.runtime.steps_per_segment)),
                phase,
                hold_joint=hold_joint,
                held_state=held_state,
                follow_attachments=follow_attachments,
                fixed_joint_states=fixed_joint_states,
                twist_target=item.get("twist_angle"),
            )
            if debug_site is not None and "target_pos" in item:
                self._settle_until_site_reached(
                    recorder,
                    np.asarray(item["action"]),
                    phase,
                    debug_site,
                    item["target_pos"],
                    hold_joint=hold_joint,
                    held_state=held_state,
                    follow_attachments=follow_attachments,
                    fixed_joint_states=fixed_joint_states,
                    twist_target=item.get("twist_angle"),
                )
                self._record_site_target_error(phase, debug_site, item["target_pos"])

    def _execute_topp_plan(
        self,
        recorder: EpisodeRecorder,
        plan: list[dict[str, Any]],
        phase_prefix: str,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        debug_site: str | None = None,
    ) -> None:
        joint_names = tuple(str(name) for name in plan[0]["arm_joint_names"])
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        action_ids = [actuator_id(self.env.model, self.env.mujoco, joint_name) for joint_name in joint_names]
        start_q = np.asarray([self.env.data.qpos[qpos_id] for qpos_id in qpos_ids], dtype=np.float64)
        q_waypoints = np.vstack([start_q, *(np.asarray(item["arm_qpos"], dtype=np.float64) for item in plan)])

        planner = Topp(ToppConfig(dof=len(joint_names), qc_vel=self.runtime.topp_vel, qc_acc=self.runtime.topp_acc))
        trajectory = planner.jnt_traj(q_waypoints)
        duration = float(trajectory.duration)
        steps = max(1, int(np.ceil(duration / self.env.control_dt)))
        segment_edges = np.linspace(0.0, duration, len(plan) + 1)

        for step in range(1, steps + 1):
            t = duration * step / steps
            q = planner.query(trajectory, t)
            item_index = int(np.searchsorted(segment_edges[1:], t, side="left"))
            item_index = min(item_index, len(plan) - 1)
            action = np.asarray(plan[item_index]["action"], dtype=np.float64).copy()
            for action_id, value in zip(action_ids, q, strict=True):
                action[action_id] = value
            phase = f"{phase_prefix}:{plan[item_index]['name']}"
            obs, *_ = self.manager.step(action)
            constrained = False
            if hold_joint is not None and held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
                constrained = True
            if follow_attachments:
                self._apply_follow_attachments(follow_attachments)
                obs = self.env.get_observation()
                constrained = True
            if fixed_joint_states:
                self._restore_fixed_joint_states(fixed_joint_states)
                obs = self.env.get_observation()
                constrained = True
            if constrained:
                obs = self._refresh_screw_pose_after_constraints(action, obs)
            recorder.record(obs, action, phase)

        final_item = plan[-1]
        if debug_site is not None and "target_pos" in final_item:
            phase = f"{phase_prefix}:{final_item['name']}"
            self._settle_until_site_reached(
                recorder,
                np.asarray(final_item["action"]),
                phase,
                debug_site,
                final_item["target_pos"],
                hold_joint=hold_joint,
                held_state=held_state,
                follow_attachments=follow_attachments,
                fixed_joint_states=fixed_joint_states,
                twist_target=final_item.get("twist_angle"),
            )
            self._record_site_target_error(phase, debug_site, final_item["target_pos"])

    def _settle_until_site_reached(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        phase: str,
        site_name: str,
        target_pos: Any,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        twist_target: float | None = None,
    ) -> None:
        target = np.asarray(target_pos, dtype=np.float64)
        max_steps = max(0, int(self.runtime.waypoint_settle_steps))
        tol = float(self.runtime.waypoint_settle_pos_tol)
        if max_steps == 0:
            return

        settle_phase = f"{phase}:settle"
        for _ in range(max_steps):
            actual, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
            if float(np.linalg.norm(target - actual)) <= tol:
                return
            if self.screw_system is not None and twist_target is not None:
                self.screw_system.set_commanded_twist(twist_target)
            obs, *_ = self.manager.step(action)
            constrained = False
            if hold_joint is not None and held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
                constrained = True
            if follow_attachments:
                self._apply_follow_attachments(follow_attachments)
                obs = self.env.get_observation()
                constrained = True
            if fixed_joint_states:
                self._restore_fixed_joint_states(fixed_joint_states)
                obs = self.env.get_observation()
                constrained = True
            if constrained:
                obs = self._refresh_screw_pose_after_constraints(action, obs)
            recorder.record(obs, action, settle_phase)

    def _record_site_target_error(self, phase: str, site_name: str, target_pos: Any) -> None:
        target = np.asarray(target_pos, dtype=np.float64)
        actual, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        error = target - actual
        entry = {
            "phase": phase,
            "site": site_name,
            "target_pos": target.tolist(),
            "actual_site_pos": actual.tolist(),
            "target_minus_actual": error.tolist(),
            "norm": float(np.linalg.norm(error)),
        }
        self.execution_site_errors.append(entry)
        err_mm = error * 1000.0
        print(
            "[site_error] "
            f"{phase} site={site_name} "
            f"target-actual(mm)=[{err_mm[0]:+.2f}, {err_mm[1]:+.2f}, {err_mm[2]:+.2f}] "
            f"norm={entry['norm'] * 1000.0:.2f}mm"
        )

    def _move_action(
        self,
        recorder: EpisodeRecorder,
        target_action: np.ndarray,
        steps: int,
        phase: str,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        twist_target: float | None = None,
    ) -> None:
        start = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        denom = max(1, steps)
        start_twist = self.screw_system.progress.twist_angle if self.screw_system is not None else 0.0
        for step in range(1, denom + 1):
            alpha = step / denom
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            action = (1.0 - alpha) * start + alpha * target_action
            if self.screw_system is not None and twist_target is not None:
                self.screw_system.set_commanded_twist(start_twist + alpha * (twist_target - start_twist))
            obs, *_ = self.manager.step(action)
            constrained = False
            if hold_joint is not None and held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
                constrained = True
            if follow_attachments:
                self._apply_follow_attachments(follow_attachments)
                obs = self.env.get_observation()
                constrained = True
            if fixed_joint_states:
                self._restore_fixed_joint_states(fixed_joint_states)
                obs = self.env.get_observation()
                constrained = True
            if constrained:
                obs = self._refresh_screw_pose_after_constraints(action, obs)
            recorder.record(obs, action, phase)
        if self.screw_system is not None and twist_target is not None:
            self.screw_system.set_commanded_twist(twist_target)

    def _hold_action(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        steps: int,
        phase: str,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
    ) -> None:
        for _ in range(max(0, steps)):
            obs, *_ = self.manager.step(action)
            constrained = False
            if hold_joint is not None and held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
                constrained = True
            if follow_attachments:
                self._apply_follow_attachments(follow_attachments)
                obs = self.env.get_observation()
                constrained = True
            if fixed_joint_states:
                self._restore_fixed_joint_states(fixed_joint_states)
                obs = self.env.get_observation()
                constrained = True
            if constrained:
                obs = self._refresh_screw_pose_after_constraints(action, obs)
            recorder.record(obs, action, phase)

    def _make_metadata(
        self,
        reset_info: dict[str, Any],
        random_info: dict[str, Any] | None,
        active_joint: str,
        active_cap_joint: str,
        active_cap_body: str,
        tube_plan: list[dict[str, Any]],
        cap_plan: list[dict[str, Any]],
        unscrew_plan: list[dict[str, Any]],
        *,
        num_steps: int,
    ) -> dict[str, Any]:
        final_obs = self.env.get_observation()
        metadata = {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(num_steps),
            "task": self.name,
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "tube_arm": self.runtime.tube_arm,
            "cap_arm": self.runtime.cap_arm,
            "active_joint": active_joint,
            "cap_joint": active_cap_joint,
            "cap_body": active_cap_body,
            "slot_index": random_info.get("slot_index") if random_info else None,
            "slot_name": random_info.get("slot_name") if random_info else None,
            "reset_info": reset_info,
            "tube_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in tube_plan],
            "cap_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in cap_plan],
            "unscrew_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in unscrew_plan],
            "execution_site_errors": json_safe(self.execution_site_errors),
            "screw_progress": {
                "released": bool(self.screw_system.progress.released if self.screw_system else False),
                "twist_angle": float(self.screw_system.progress.twist_angle if self.screw_system else 0.0),
                "lift_distance": float(self.screw_system.progress.lift_distance if self.screw_system else 0.0),
                "release_angle_target": float(self.runtime.release_angle),
            },
            "final_time": float(final_obs["time"]),
            "final_state_summary": json_safe(
                {
                    "tube_pos": free_joint_pos(self.env.model, self.env.data, self.env.mujoco, active_joint),
                    "cap_pos": body_pos(self.env.model, self.env.data, self.env.mujoco, active_cap_body),
                }
            ),
        }
        return metadata
