# AutoLabSim

AutoLabSim is a MuJoCo-based simulation project for laboratory manipulation tasks. The current codebase focuses on tube grasping, cap manipulation, and episode generation/export workflows.

## Project Structure

### 1. Scene Models

- `model/scenes/scene_mujoco.xml`
  Main scene entry. Defines robot arms, table, rack, cameras, and task objects.
- `model/scenes/scene_mujoco_fast_tubes.xml`
  Lightweight scene for faster and more stable data generation.

### 2. Submodels and Assets

- `model/models/grippers/2f85.xml`
  Standard gripper model.
- `model/models/grippers/2f85_fast.xml`
  Simplified collision version for fast simulation.
- `assets/`
  Mesh assets such as robot, container, rack, tool, incubator, and workbench geometry.

### 3. Reset Configuration

This layer controls object placement and randomized initialization at the start of each episode.

- `configs/reset_default.json`
  General-purpose default reset configuration.
- `configs/reset_single_tube_random.json`
  Current single-tube task reset configuration. It defines robot initial joints, the active tube, cap-tube pairing, randomized slot placement, and how inactive objects are hidden.

### 4. Task Logic

- `autolabsim/tasks/tube_grasp.py`
  Single-arm tube grasp task.
- `autolabsim/tasks/screw_cap.py`
  Dual-arm cap unscrewing task.
- `autolabsim/screw.py`
  Shared screw/unscrew logic.
- `autolabsim/tasks/__init__.py`
  Task registry used to create tasks by name.

### 5. Simulation Runtime

- `autolabsim/mujoco_env.py`
  MuJoCo environment wrapper.
- `autolabsim/scene.py`
  Scene access helpers for joints, sites, and bodies.
- `autolabsim/reset_config.py`
  Reset configuration loading and application.
- `autolabsim/scene_profile.py`
  Default scene/reset/camera profile management for each task.

### 6. Data Recording and Scripts

- `scripts/generate_tube_grasp_batch.py`
  Batch episode generation entry point.
- `scripts/visualize_task_points.py`
  Visualize grasp, pre-grasp, and lift points.
- `scripts/view_episode.py`
  Replay saved episodes.
- `scripts/export_episode_images.py`
  Export episode images or videos.
- `autolabsim/recorder.py`
  Record `qpos`, `qvel`, `ctrl`, and image streams.
- `autolabsim/episode_io.py`
  Episode read/write utilities.

## Reset Template

If you want to add a new task scene, a good first step is usually to create a new reset configuration instead of changing task code immediately.

```bash
cp configs/templates/reset_scene_template.json configs/reset_my_task.json
```

Key fields in the template:

- `actuators`
  Initial robot and gripper control targets. Names must match the actuators in the XML.
- `free_joints`
  Initial poses for free objects. Useful for hiding unused objects or assigning fixed poses.
- `random_single_free_joint`
  A common initialization pattern: choose one `active_joint`, place it into a random slot, and move the others to `inactive_pose`.

Recommended checklist:

1. `joints`
   List all candidate free-joint names. They must exist in the scene XML.
2. `active_joint`
   Define which object is treated as the main object by the task logic.
3. `companion_joints`
   Describe paired objects such as a tube and its cap. `pos_offset` gives the relative position offset.
4. `slots`
   Define the candidate placement pool. Each slot should include at least `name`, `pos`, and `quat`.

Minimal rules:

- Names must match the XML exactly.
- JSON structure should follow the template.
- Semantics should stay consistent with the task logic.

## Common Commands

### View a Scene with Reset Randomization

```bash
python scripts/view_scene.py model/scenes/scene_mujoco_fast_tubes.xml \
  --reset-config configs/reset_single_tube_random.json
```

### Generate Episodes with Images

```bash
python scripts/generate_tube_grasp_batch.py \
  --scene fast_tubes \
  --task tube_then_cap_grasp \
  --count 1 \
  --seed-start 40 \
  --with-images \
  --cameras overview_camera,wrist_cam,wrist_cam1 \
  --out-root data/episodes/bimanual_unscrew_video_test
```

### Export Episode Video

```bash
python scripts/export_episode_images.py \
  data/episodes/bimanual_unscrew_video_test \
  --camera overview_camera,wrist_cam,wrist_cam1 \
  --format mp4 \
  --fps 20
```

## GitHub Upload and Sync

This repository currently uses:

- branch: `main`
- remote: `origin -> https://github.com/bumingli2000-stack/AutoLabSim.git`

### First Upload

If the repository has not been pushed successfully before:

```bash
cd /home/buming/projects/AutoLabSim
git status
git add .
git commit -m "Initial commit"
git push -u origin main
```

If your terminal cannot access GitHub directly and you use Clash, you may need:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

### Daily Update Workflow

After editing local files:

```bash
cd /home/buming/projects/AutoLabSim
git status
git add .
git commit -m "Describe your changes"
git push
```

This does not delete the old version. Git creates a new commit on top of the previous history, so earlier versions remain available.

### Check History

```bash
git log --oneline --graph --decorate -n 10
```

### Pull Remote Updates First

If you also changed files on GitHub or another machine pushed updates before you:

```bash
git pull --rebase origin main
git push
```

Using `pull --rebase` keeps the history cleaner and reduces unnecessary merge commits.

## Multi-Developer Collaboration

If multiple people will work on this project, the safest habit is:

### 1. Always Pull Before Starting Work

```bash
git pull --rebase origin main
```

This reduces the chance of developing on an outdated local copy.

### 2. Keep Each Change Small and Clear

Use focused commit messages, for example:

```bash
git commit -m "Add screw cap reset config"
git commit -m "Fix tube grasp point generation"
```

Small commits are easier to review, merge, and revert.

### 3. Prefer Branches for New Features

For non-trivial work, create a feature branch:

```bash
git checkout -b feature/export-lerobot-data
```

After finishing:

```bash
git add .
git commit -m "Add LeRobot export script"
git push -u origin feature/export-lerobot-data
```

Then open a Pull Request on GitHub instead of pushing everything directly to `main`.

### 4. Protect `main`

For team development, treat `main` as the stable branch:

- only merge reviewed changes into `main`
- avoid direct force-push to `main`
- avoid mixing unrelated edits in one commit

### 5. Resolve Conflicts Explicitly

If `git pull --rebase` reports conflicts:

```bash
git status
```

Open the conflicting files, resolve the marked sections, then continue:

```bash
git add <resolved-file>
git rebase --continue
```

### 6. Keep Large Data Out of Git

This repository already ignores:

- `.vscode/`
- `data/`
- `MUJOCO_LOG.TXT`

That is important for collaboration because generated data and local IDE files should not be committed to the shared repository. If you later need to share large datasets, use a separate storage solution or Git LFS.
