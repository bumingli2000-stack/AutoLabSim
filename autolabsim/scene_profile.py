from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneNaming:
    active_joint_fallback: str
    tube_joint_prefix: str
    cap_joint_prefix: str
    cap_body_prefix: str
    cap_weld_prefix: str


@dataclass(frozen=True)
class SceneSpec:
    name: str
    model_path: str
    reset_config: str | None
    cameras: tuple[str, ...]
    naming: SceneNaming


DEFAULT_SCENE_NAMING = SceneNaming(
    active_joint_fallback="centrifuge_50ml_screw_joint_1",
    tube_joint_prefix="centrifuge_50ml_screw_joint_",
    cap_joint_prefix="centrifuge_50ml_screw_cap_joint_",
    cap_body_prefix="centrifuge_50ml_screw_cap_",
    cap_weld_prefix="centrifuge_50ml_screw_cap_weld_",
)


SCENE_REGISTRY: dict[str, SceneSpec] = {
    "fast_tubes": SceneSpec(
        name="fast_tubes",
        model_path="model/scenes/scene_mujoco_fast_tubes.xml",
        reset_config="configs/reset_single_tube_random.json",
        cameras=("overview_camera",),
        naming=DEFAULT_SCENE_NAMING,
    ),
    "default": SceneSpec(
        name="default",
        model_path="model/scenes/scene_mujoco.xml",
        reset_config="configs/reset_default.json",
        cameras=("overview_camera",),
        naming=DEFAULT_SCENE_NAMING,
    ),
}


def scene_names() -> tuple[str, ...]:
    return tuple(SCENE_REGISTRY.keys())


def get_scene_spec(name: str) -> SceneSpec:
    try:
        return SCENE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown scene profile: {name}") from exc


def resolve_scene_spec(
    scene_name: str,
    *,
    model: str | None = None,
    reset_config: str | None = None,
    cameras: tuple[str, ...] | None = None,
) -> SceneSpec:
    base = get_scene_spec(scene_name)
    return SceneSpec(
        name=base.name,
        model_path=model or base.model_path,
        reset_config=reset_config if reset_config is not None else base.reset_config,
        cameras=cameras or base.cameras,
        naming=base.naming,
    )


def active_joint_fallback(scene_name: str = "fast_tubes") -> str:
    return get_scene_spec(scene_name).naming.active_joint_fallback


def scene_rooted_path(scene_spec: SceneSpec, root: Path) -> tuple[Path, Path | None]:
    model_path = Path(scene_spec.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path

    reset_path: Path | None = None
    if scene_spec.reset_config is not None:
        reset_path = Path(scene_spec.reset_config)
        if not reset_path.is_absolute():
            reset_path = root / reset_path
    return model_path, reset_path
