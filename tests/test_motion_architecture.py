from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np

from autolabsim.executor import TaskTargetExecutor
from autolabsim.motion_context import (
    ArmMotionConfig,
    ExecutionSettings,
    GripperSettings,
    IKSettings,
    KinematicBinding,
    PlanningContext,
    SiteAttachment,
    VisualServoSettings,
)
from autolabsim.mujoco_env import EnvConfig
from autolabsim.planner import TaskTargetPlanner
from autolabsim.task_target import (
    FrameRef,
    GripperCommand,
    PlannedTaskTarget,
    PoseOffset,
    ResolvedTaskTarget,
    TaskTarget,
    TaskTargetResolver,
    _mat_to_quat_numpy,
)
from AutoLabSim.autolabsim.tasks.pipette_grasp.pipette_grasp import PipetteGraspTaskConfig


IDENTITY_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class FakeObj:
    mjOBJ_BODY = 1
    mjOBJ_SITE = 2
    mjOBJ_JOINT = 3
    mjOBJ_ACTUATOR = 4


class FakeJoint:
    mjJNT_FREE = 0


class FakeMujoco:
    mjtObj = FakeObj
    mjtJoint = FakeJoint

    def __init__(
        self,
        *,
        sites: tuple[str, ...] = (),
        joints: tuple[str, ...] = (),
        bodies: tuple[str, ...] = (),
        actuators: tuple[str, ...] = (),
    ):
        self.site_ids = {name: index for index, name in enumerate(sites)}
        self.joint_ids = {name: index for index, name in enumerate(joints)}
        self.body_ids = {name: index for index, name in enumerate(bodies)}
        self.actuator_ids = {name: index for index, name in enumerate(actuators)}

    def mj_name2id(self, model: Any, obj_type: int, name: str) -> int:
        del model
        maps = {
            self.mjtObj.mjOBJ_BODY: self.body_ids,
            self.mjtObj.mjOBJ_SITE: self.site_ids,
            self.mjtObj.mjOBJ_JOINT: self.joint_ids,
            self.mjtObj.mjOBJ_ACTUATOR: self.actuator_ids,
        }
        return maps[obj_type].get(name, -1)

    def mju_mat2Quat(self, out: np.ndarray, mat: np.ndarray) -> None:
        out[:] = _mat_to_quat_numpy(np.asarray(mat, dtype=np.float64).reshape(3, 3))

    def mj_forward(self, model: Any, data: Any) -> None:
        del model, data


class FakeModel:
    def __init__(self, joint_count: int = 0, actuator_count: int = 0):
        self.jnt_qposadr = np.asarray([7 * index for index in range(joint_count)], dtype=np.int32)
        self.jnt_dofadr = np.asarray([6 * index for index in range(joint_count)], dtype=np.int32)
        self.jnt_type = np.zeros(joint_count, dtype=np.int32)
        self.jnt_limited = np.zeros(joint_count, dtype=np.int32)
        self.jnt_range = np.zeros((joint_count, 2), dtype=np.float64)
        self.nu = actuator_count


class FakeData:
    def __init__(
        self,
        *,
        joint_poses: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
        site_poses: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
        body_poses: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
        ctrl_size: int = 0,
    ):
        self.qpos = np.zeros(max(1, 7 * len(joint_poses)), dtype=np.float64)
        self.qvel = np.zeros(max(1, 6 * len(joint_poses)), dtype=np.float64)
        for index, (pos, quat) in enumerate(joint_poses):
            adr = 7 * index
            self.qpos[adr : adr + 3] = pos
            self.qpos[adr + 3 : adr + 7] = quat
        self.ctrl = np.zeros(max(1, ctrl_size), dtype=np.float64)
        self.site_xpos = np.asarray([pos for pos, _ in site_poses], dtype=np.float64)
        self.site_xmat = np.asarray([np.eye(3, dtype=np.float64).reshape(-1) for _ in site_poses])
        self.xpos = np.asarray([pos for pos, _ in body_poses], dtype=np.float64)
        self.xmat = np.asarray([np.eye(3, dtype=np.float64).reshape(-1) for _ in body_poses])


class FakeEnv:
    def __init__(self, model: FakeModel, data: FakeData, mujoco: FakeMujoco):
        self.model = model
        self.data = data
        self.mujoco = mujoco

    def get_observation(self) -> dict[str, Any]:
        return {"ctrl": self.data.ctrl.copy(), "time": 0.0}


class FakeManager:
    def __init__(self, env: FakeEnv):
        self.env = env

    def step(self, action: np.ndarray) -> tuple[dict[str, Any]]:
        self.env.data.ctrl[:] = action
        return (self.env.get_observation(),)


class FakeRecorder:
    def __init__(self):
        self.actions: list[np.ndarray] = []
        self.phases: list[str] = []

    def record(self, obs: dict[str, Any], action: np.ndarray, phase: str) -> None:
        del obs
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.phases.append(phase)


def planner_fixture() -> tuple[TaskTargetPlanner, PlanningContext]:
    mujoco = FakeMujoco(
        sites=("gripper", "tool_site", "piptip_site", "tip_site"),
        joints=("tool_joint", "tip_joint"),
    )
    model = FakeModel(joint_count=2)
    data = FakeData(
        joint_poses=(
            (np.asarray([1.0, 0.0, 0.0]), IDENTITY_QUAT),
            (np.asarray([1.0, 0.0, 2.0]), IDENTITY_QUAT),
        ),
        site_poses=(
            (np.asarray([0.0, 0.0, 0.0]), IDENTITY_QUAT),
            (np.asarray([1.0, 0.0, 1.0]), IDENTITY_QUAT),
            (np.asarray([1.0, 0.0, 1.0]), IDENTITY_QUAT),
            (np.asarray([1.0, 0.0, 3.0]), IDENTITY_QUAT),
        ),
    )
    planner = TaskTargetPlanner(
        model,
        data,
        mujoco,
        {"first": ArmMotionConfig("first", ("j0",), "gripper", "grip")},
        IKSettings(),
        GripperSettings(0.0, 255.0),
    )
    one_level = SiteAttachment(
        joint_name="tool_joint",
        parent_site="gripper",
        local_pos=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        local_quat=IDENTITY_QUAT,
    )
    mounted_tip = SiteAttachment(
        joint_name="tip_joint",
        parent_site="piptip_site",
        local_pos=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        local_quat=IDENTITY_QUAT,
    )
    context = PlanningContext(
        (
            KinematicBinding("first", "gripper", "tool_site", (one_level,)),
            KinematicBinding("first", "gripper", "tip_site", (one_level, mounted_tip)),
        )
    )
    return planner, context


class MotionArchitectureTest(unittest.TestCase):
    def test_body_target_offset_composes_body_grasp_gripper(self) -> None:
        mujoco = FakeMujoco(bodies=("pipette",))
        model = FakeModel()
        data = FakeData(
            body_poses=((np.asarray([1.0, 2.0, 3.0]), IDENTITY_QUAT),),
        )
        resolver = TaskTargetResolver(model, data, mujoco)
        target = TaskTarget(
            name="grasp",
            parent=FrameRef("body", "pipette"),
            pos=(0.2, 0.0, 0.0),
            euler=(0.0, 0.0, np.pi / 2.0),
            target_offset=PoseOffset(pos=(0.0, 1.0, 0.0)),
        )

        resolved = resolver.resolve(target)

        np.testing.assert_allclose(resolved.pos, np.asarray([0.2, 2.0, 3.0]), atol=1e-9)

    def test_local_approach_offset_is_composed_after_target_offset(self) -> None:
        mujoco = FakeMujoco()
        model = FakeModel()
        data = FakeData()
        resolver = TaskTargetResolver(model, data, mujoco)
        target = TaskTarget(
            name="grasp",
            parent=FrameRef("world"),
            pos=(0.0, 0.0, 0.0),
            target_offset=PoseOffset(pos=(1.0, 0.0, 0.0)),
        )

        pregrasp = target.with_approach_offset(-0.5, name="pregrasp")
        resolved = resolver.resolve(pregrasp)

        self.assertEqual(pregrasp.name, "pregrasp")
        np.testing.assert_allclose(resolved.pos, np.asarray([1.0, 0.0, -0.5]), atol=1e-9)

    def test_planner_uses_gripper_site_without_binding(self) -> None:
        planner, context = planner_fixture()
        target = TaskTarget(
            name="direct",
            parent=FrameRef("world"),
            pos=(0.4, 0.5, 0.6),
            arm="first",
            controlled_site="gripper",
        )

        _, ik_pos, ik_quat, binding = planner.target_to_ik_target(target, context)

        self.assertEqual(binding.controlled_site, "gripper")
        np.testing.assert_allclose(ik_pos, np.asarray([0.4, 0.5, 0.6]), atol=1e-9)
        np.testing.assert_allclose(ik_quat, IDENTITY_QUAT, atol=1e-9)

    def test_planner_resolves_one_level_attachment_chain(self) -> None:
        planner, context = planner_fixture()
        target = TaskTarget(
            name="tool_target",
            parent=FrameRef("world"),
            pos=(2.0, 0.0, 1.0),
            arm="first",
            controlled_site="tool_site",
        )

        _, ik_pos, ik_quat, _ = planner.target_to_ik_target(target, context)

        np.testing.assert_allclose(ik_pos, np.asarray([1.0, 0.0, 0.0]), atol=1e-9)
        np.testing.assert_allclose(ik_quat, IDENTITY_QUAT, atol=1e-9)

    def test_planner_resolves_multi_level_attachment_chain(self) -> None:
        planner, context = planner_fixture()
        target = TaskTarget(
            name="tip_target",
            parent=FrameRef("world"),
            pos=(2.0, 0.0, 3.0),
            arm="first",
            controlled_site="tip_site",
        )

        _, ik_pos, ik_quat, _ = planner.target_to_ik_target(target, context)

        np.testing.assert_allclose(ik_pos, np.asarray([1.0, 0.0, 0.0]), atol=1e-9)
        np.testing.assert_allclose(ik_quat, IDENTITY_QUAT, atol=1e-9)

    def test_executor_applies_gripper_before_command(self) -> None:
        mujoco = FakeMujoco(actuators=("grip",))
        model = FakeModel(actuator_count=1)
        data = FakeData(ctrl_size=1)
        env = FakeEnv(model, data, mujoco)
        executor = TaskTargetExecutor(
            env,
            FakeManager(env),
            {"first": ArmMotionConfig("first", ("j0",), "gripper", "grip")},
            IKSettings(),
            GripperSettings(0.0, 10.0),
            ExecutionSettings(visual_servo=VisualServoSettings(enabled=False)),
        )
        target = TaskTarget(
            name="command",
            parent=FrameRef("world"),
            pos=(0.0, 0.0, 0.0),
            arm="first",
            controlled_site="gripper",
            gripper=GripperCommand(255.0, timing="before", steps=2),
        )
        resolved = ResolvedTaskTarget(target, np.zeros(3), IDENTITY_QUAT, np.eye(3))
        item = PlannedTaskTarget(
            target=target,
            resolved=resolved,
            ik_site_pos=np.zeros(3),
            ik_site_quat=IDENTITY_QUAT,
            action=np.zeros(1),
            ik_success=True,
            ik_pos_error=0.0,
            ik_rot_error=0.0,
            arm_joint_names=("j0",),
            arm_qpos=np.zeros(1),
        )
        recorder = FakeRecorder()

        executor.apply_target_gripper_command(recorder, item, "before", "phase")

        self.assertEqual(len(recorder.actions), 2)
        self.assertAlmostEqual(float(recorder.actions[-1][0]), 10.0)
        self.assertAlmostEqual(float(item.action[0]), 10.0)

    def test_executor_dispatches_position_and_pose_servo(self) -> None:
        class DispatchExecutor(TaskTargetExecutor):
            def __init__(self) -> None:
                mujoco = FakeMujoco()
                model = FakeModel()
                data = FakeData()
                env = FakeEnv(model, data, mujoco)
                super().__init__(
                    env,
                    FakeManager(env),
                    {"first": ArmMotionConfig("first", ("j0",), "gripper", "grip")},
                    IKSettings(),
                    GripperSettings(0.0, 255.0),
                    ExecutionSettings(visual_servo=VisualServoSettings(enabled=True)),
                )
                self.calls: list[tuple[str, bool]] = []

            def move_action(self, recorder: Any, target_action: np.ndarray, steps: int, phase: str, context: Any = None) -> None:
                del recorder, target_action, steps, phase, context

            def visual_servo_site_to_target(
                self,
                recorder: Any,
                action: np.ndarray,
                phase: str,
                site_name: str,
                target_pos: Any,
                target_quat: Any | None = None,
                *,
                arm_name: str,
                context: Any = None,
            ) -> np.ndarray:
                del recorder, phase, target_pos, arm_name, context
                self.calls.append((site_name, target_quat is not None))
                return action

            def record_site_target_error(self, phase: str, site_name: str, target_pos: Any) -> None:
                del phase, site_name, target_pos

        def planned(name: str, servo_mode: str) -> PlannedTaskTarget:
            target = TaskTarget(
                name=name,
                parent=FrameRef("world"),
                pos=(0.0, 0.0, 0.0),
                arm="first",
                controlled_site="gripper",
                servo_mode=servo_mode,
            )
            return PlannedTaskTarget(
                target=target,
                resolved=ResolvedTaskTarget(target, np.zeros(3), IDENTITY_QUAT, np.eye(3)),
                ik_site_pos=np.zeros(3),
                ik_site_quat=IDENTITY_QUAT,
                action=np.zeros(1),
                ik_success=True,
                ik_pos_error=0.0,
                ik_rot_error=0.0,
                arm_joint_names=("j0",),
                arm_qpos=np.zeros(1),
            )

        executor = DispatchExecutor()
        executor.execute(FakeRecorder(), [planned("position", "position"), planned("pose", "pose")], "phase")

        self.assertEqual(executor.calls, [("gripper", False), ("gripper", True)])

    def test_pipette_config_uses_nested_access_only(self) -> None:
        config = PipetteGraspTaskConfig(
            env=EnvConfig(),
            out_dir=Path("/tmp/autolabsim-test"),
            episode_index=0,
            seed=1,
        )

        self.assertEqual(config.robot.arm, "first")
        self.assertEqual(config.grasp.handle_grasp_offset, (0.0, 0.0, 0.15))
        self.assertFalse(hasattr(config, "arm"))
        self.assertFalse(hasattr(config, "ik_max_iters"))


if __name__ == "__main__":
    unittest.main()
