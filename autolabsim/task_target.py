'''
任务目标点的统一表示和坐标系解析。
'''
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .math3d import (
    euler_xyz_to_mat,
    mat_to_quat,
    normalize_quat,
    quat_conjugate,
    quat_multiply,
    quat_to_mat,
)
from .scene import free_joint_pose, site_pose

# =============================================================================
# 目标点坐标系引用，主要用于指定目标点的父坐标系。
    # 父坐标系的种类包括：
    #     - world: 世界坐标系
    #     - site: 指定的 site 坐标系
    #     - free_joint: 指定的自由关节坐标系
    #     - body: 指定的 body 坐标系    
    # name: 指定的坐标系名称，必须与 kind 对应。  
# =============================================================================
@dataclass(frozen=True)
class FrameRef:
    kind: str = "world"
    name: str | None = None

# =============================================================================
# 夹爪命令，主要用于指定夹爪的开合状态，执行时机，执行步数。
    # value：开合状态的取值范围为 0.0 ~ 255.0，255.0 表示完全闭合，0.0 表示完全张开。
    # timing：执行时机的取值范围为 "before"、"during"、"after"，分别表示在目标点到达前、到达时、到达后执行夹爪命令。
    # steps：执行步数的取值范围为 1 ~ 100，表示夹爪命令的执行步数，默认值为 12。
# =============================================================================
@dataclass(frozen=True)
class GripperCommand:
    value: float
    timing: str = "during"
    steps: int = 12

@dataclass(frozen=True)
class PoseOffset:
    """Local pose offset composed after a target pose."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] | None = None


# =============================================================================
# 任务目标点，主要用于指定机器人末端执行器的目标位置和姿态，以及夹爪的开合状态。
    # name: 目标点的名称，必须唯一。
    # parent: 目标点的父坐标系引用，必须为 FrameRef 类型。
    # pos: 目标点在父坐标系下的相对位置，必须为三维向量。
    # euler: 目标点在父坐标系下的相对欧拉角，必须为三维向量，默认值为 (0.0, 0.0, 0.0)。
    # quat_wxyz: 目标点在父坐标系下的相对四元数，必须为四维向量，默认值为 None。
    # arm: 目标点对应的机械臂名称，必须为字符串，默认值为 "first"。
    # controlled_site: 目标点对应的受控 site 名称，必须为字符串，默认值为 ""。
    # servo_mode: 目标点的伺服模式，必须为字符串，取值范围为 "none"、"pose"、"position"，分别表示不伺服、伺服到位姿、伺服到位置，默认值为 "none" 
# =============================================================================
@dataclass(frozen=True)
class TaskTarget:
    name: str
    parent: FrameRef
    pos: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] | None = None
    arm: str = "first"
    controlled_site: str = ""
    servo_mode: str = "none"
    gripper: GripperCommand | None = None
    target_offset: PoseOffset | None = None

    def offset_local(
        self,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat_wxyz: tuple[float, float, float, float] | None = None,
        name: str | None = None,
    ) -> "TaskTarget":
        """Return a target shifted in the final controlled-site local frame."""

        extra = PoseOffset(translation, euler, quat_wxyz)
        offset = compose_offsets(self.target_offset, extra)
        return replace(self, name=name or self.name, target_offset=offset)

    def with_approach_offset(self, distance: float, *, name: str | None = None) -> "TaskTarget":
        """Offset along the target's local +Z approach axis."""

        return self.offset_local((0.0, 0.0, float(distance)), name=name)

# =============================================================================
# 解析后的任务目标点，主要用于存储任务目标点在世界坐标系下的绝对位置和姿态。
    # spec: 任务目标点的原始规格，必须为 TaskTarget 类型。
    # pos: 任务目标点在世界坐标系下的绝对位置，必须为三维向量。
    # quat: 任务目标点在世界坐标系下的绝对四元数，必须为四维向量。
    # mat: 任务目标点在世界坐标系下的绝对旋转矩阵，必须为 3x3 矩阵。
# =============================================================================
@dataclass(frozen=True)
class ResolvedTaskTarget:
    spec: TaskTarget
    pos: np.ndarray
    quat: np.ndarray
    mat: np.ndarray

# =============================================================================
# 计划的任务目标点，主要用于存储任务目标点在世界坐标系下的绝对位置和姿态，以及逆运动学求解结果和机械臂关节状态。
    # target: 任务目标点的原始规格，必须为 TaskTarget 类型。
    # resolved: 任务目标点的解析结果，必须为 ResolvedTaskTarget 类型。
    # ik_site_pos: 逆运动学求解得到的机械臂末端执行器在世界坐标系下的绝对位置，必须为三维向量。 
    # ik_site_quat: 逆运动学求解得到的机械臂末端执行器在世界坐标系下的绝对四元数，必须为四维向量。
    # action: 逆运动学求解得到的机械臂关节动作，必须为一维向量。
    # ik_success: 逆运动学求解是否成功，必须为布尔值。
    # ik_pos_error: 逆运动学求解得到的机械臂末端执行器位置误差，必须为浮点数。
    # ik_rot_error: 逆运动学求解得到的机械臂末端执行器姿态误差，必须为浮点数。
    # arm_joint_names: 机械臂关节名称，必须为字符串元组。
    # arm_qpos: 机械臂关节位置，必须为一维向量。
    # gripper_value: 夹爪执行器值，必须为浮点数或 None。
    # steps: 任务目标点的执行步数，必须为整数或 None。
    # twist_angle: 任务目标点的扭转角度，必须为浮点数或 None。
    # extra: 任务目标点的额外信息，必须为字典或 None。
# =============================================================================
@dataclass
class PlannedTaskTarget:
    target: TaskTarget
    resolved: ResolvedTaskTarget
    ik_site_pos: np.ndarray
    ik_site_quat: np.ndarray
    action: np.ndarray
    ik_success: bool
    ik_pos_error: float
    ik_rot_error: float
    arm_joint_names: tuple[str, ...]
    arm_qpos: np.ndarray
    gripper_value: float | None = None
    steps: int | None = None
    twist_angle: float | None = None
    extra: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def servo_mode(self) -> str:
        return self.target.servo_mode

    @property
    def debug_target_pos(self) -> np.ndarray | None:
        return None if self.servo_mode == "none" else self.resolved.pos

    @property
    def debug_target_quat(self) -> np.ndarray | None:
        return self.resolved.quat if self.servo_mode == "pose" else None

    @property
    def gripper_command(self) -> GripperCommand | None:
        return self.target.gripper

    def put_extra(self, key: str, value: Any) -> None:
        if self.extra is None:
            self.extra = {}
        self.extra[key] = value

    def extra_value(self, key: str, default: Any = None) -> Any:
        if self.extra is None:
            return default
        return self.extra.get(key, default)

    # =============================================================================
    # 将计划的任务目标点转换为元数据字典，主要用于记录任务目标点的相关信息，包括名称、位置、姿态、逆运动学求解结果、机械臂关节状态、夹爪命令等。
    # ============================================================================= 
    def to_metadata(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "name": self.name,
            "target_pos": np.asarray(self.ik_site_pos, dtype=np.float64).tolist(),
            "target_quat_wxyz": np.asarray(self.ik_site_quat, dtype=np.float64).tolist(),
            "ik_success": bool(self.ik_success),
            "ik_pos_error": float(self.ik_pos_error),
            "ik_rot_error": float(self.ik_rot_error),
            "arm_joint_names": list(self.arm_joint_names),
            "arm_qpos": np.asarray(self.arm_qpos, dtype=np.float64).tolist(),
            "task_target": task_target_metadata(
                self.target,
                self.resolved,
                gripper_actuator_value=self.gripper_value,
            ),
            "servo_mode": self.servo_mode,
        }
        if self.debug_target_pos is not None:
            item["debug_target_pos"] = self.debug_target_pos.tolist()
        if self.debug_target_quat is not None:
            item["debug_target_quat"] = self.debug_target_quat.tolist()
        if self.gripper_command is not None:
            item["gripper_command"] = asdict(self.gripper_command)
            item["gripper_value"] = self.gripper_value
        if self.steps is not None:
            item["steps"] = int(self.steps)
        if self.twist_angle is not None:
            item["twist_angle"] = float(self.twist_angle)
        if self.extra:
            item.update(self.extra)
        return item

# =============================================================================
# 目标点坐标系解析类，主要负责把对应目标点的 parent + pos/euler 解析成世界坐标。
# =============================================================================
class TaskTargetResolver:
    def __init__(self, model: Any, data: Any, mujoco: Any):
        self.model = model
        self.data = data
        self.mujoco = mujoco

    def resolve_frame_ref(self, frame: FrameRef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if frame.kind == "world":
            mat = np.eye(3, dtype=np.float64)
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return np.zeros(3, dtype=np.float64), quat, mat
        if not frame.name:
            raise ValueError(f"FrameRef kind {frame.kind!r} requires a name")
        if frame.kind == "site":
            pos, quat = site_pose(self.model, self.data, self.mujoco, frame.name)
            return pos, quat, quat_to_mat(quat)
        if frame.kind == "free_joint":
            pos, quat = free_joint_pose(self.model, self.data, self.mujoco, frame.name)
            return pos, quat, quat_to_mat(quat)
        if frame.kind == "body":
            body_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_BODY,
                frame.name,
            )
            if body_id < 0:
                raise ValueError(f"Unknown body frame: {frame.name}")
            pos = np.asarray(self.data.xpos[body_id], dtype=np.float64).copy()
            mat = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
            return pos, mat_to_quat(self.mujoco, mat), mat
        raise ValueError(f"Unknown frame kind: {frame.kind}")

    def resolve(self, target: TaskTarget) -> ResolvedTaskTarget:
        parent_pos, _, parent_mat = self.resolve_frame_ref(target.parent)
        local_pos = np.asarray(target.pos, dtype=np.float64)
        local_mat = (
            quat_to_mat(np.asarray(target.quat_wxyz, dtype=np.float64))
            if target.quat_wxyz is not None
            else euler_xyz_to_mat(np.asarray(target.euler, dtype=np.float64))
        )
        if target.target_offset is not None:
            offset_pos, offset_mat = pose_offset_arrays(target.target_offset)
            local_pos = local_pos + local_mat @ offset_pos
            local_mat = local_mat @ offset_mat
        mat = parent_mat @ local_mat
        pos = parent_pos + parent_mat @ local_pos
        return ResolvedTaskTarget(target, pos, mat_to_quat(self.mujoco, mat), mat)

    def relative_quat_for_world_quat(self, parent: FrameRef, world_quat: np.ndarray) -> np.ndarray:
        _, parent_quat, _ = self.resolve_frame_ref(parent)
        return normalize_quat(quat_multiply(quat_conjugate(parent_quat), world_quat))


def gripper_command_to_actuator(command: GripperCommand, open_value: float, close_value: float) -> float:
    value_255 = min(255.0, max(0.0, float(command.value)))
    alpha = value_255 / 255.0
    return float(open_value + (close_value - open_value) * alpha)


def pose_offset_arrays(offset: PoseOffset) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(offset.pos, dtype=np.float64)
    mat = (
        quat_to_mat(np.asarray(offset.quat_wxyz, dtype=np.float64))
        if offset.quat_wxyz is not None
        else euler_xyz_to_mat(np.asarray(offset.euler, dtype=np.float64))
    )
    return pos, mat


def pose_offset_from_arrays(mujoco: Any, pos: np.ndarray, mat: np.ndarray) -> PoseOffset:
    return PoseOffset(
        pos=tuple(np.asarray(pos, dtype=np.float64).tolist()),
        quat_wxyz=tuple(mat_to_quat(mujoco, mat).tolist()),
    )


def compose_offsets(first: PoseOffset | None, second: PoseOffset | None, mujoco: Any | None = None) -> PoseOffset | None:
    if first is None:
        return second
    if second is None:
        return first
    first_pos, first_mat = pose_offset_arrays(first)
    second_pos, second_mat = pose_offset_arrays(second)
    pos = first_pos + first_mat @ second_pos
    mat = first_mat @ second_mat
    if mujoco is None:
        # Keep the composed pose as a matrix-derived quaternion when a MuJoCo
        # converter is not available. This branch is used for local target
        # transformations before resolver-time metadata is needed.
        quat = _mat_to_quat_numpy(mat)
        return PoseOffset(pos=tuple(pos.tolist()), quat_wxyz=tuple(quat.tolist()))
    return pose_offset_from_arrays(mujoco, pos, mat)


def _mat_to_quat_numpy(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0) * 2.0)
        quat = np.asarray(
            [
                0.25 * scale,
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            scale = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0)
            quat = np.asarray(
                [
                    (m[2, 1] - m[1, 2]) / scale,
                    0.25 * scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            scale = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0)
            quat = np.asarray(
                [
                    (m[0, 2] - m[2, 0]) / scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    0.25 * scale,
                    (m[1, 2] + m[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0)
            quat = np.asarray(
                [
                    (m[1, 0] - m[0, 1]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return normalize_quat(quat)

# =============================================================================
# 将任务目标点转换为元数据字典，主要用于记录任务目标点的相关信息，包括名称、位置、姿态、夹爪命令等。
# =============================================================================
def task_target_metadata(
    target: TaskTarget,
    resolved: ResolvedTaskTarget,
    *,
    gripper_actuator_value: float | None = None,
) -> dict[str, Any]:
    gripper = None
    if target.gripper is not None:
        gripper = asdict(target.gripper)
        if gripper_actuator_value is not None:
            gripper["actuator_value"] = gripper_actuator_value
    target_offset = None
    if target.target_offset is not None:
        target_offset = {
            "pos": list(target.target_offset.pos),
            "euler_xyz": list(target.target_offset.euler),
            "quat_wxyz": list(target.target_offset.quat_wxyz)
            if target.target_offset.quat_wxyz is not None
            else None,
        }
    return {
        "name": target.name,
        "parent": {"kind": target.parent.kind, "name": target.parent.name},
        "relative_pos": list(target.pos),
        "relative_euler_xyz": list(target.euler),
        "relative_quat_wxyz": list(target.quat_wxyz) if target.quat_wxyz is not None else None,
        "target_offset": target_offset,
        "arm": target.arm,
        "controlled_site": target.controlled_site,
        "servo_mode": target.servo_mode,
        "gripper": gripper,
        "world_pos": resolved.pos.tolist(),
        "world_quat_wxyz": resolved.quat.tolist(),
    }
