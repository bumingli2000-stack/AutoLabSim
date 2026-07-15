# AutoLabSim ACT training add-on

This add-on trains a compact CVAE Action Chunking Transformer directly from
AutoLabSim's native `metadata.json + episode.npz` episodes.

## Important alignment rule

AutoLabSim records each observation **after** `env.step(action)`. Therefore the
same-index `ctrl[t]` and `action[t]` are normally nearly identical. The default
training alignment is:

```text
observation[t] -> action[t+1 : t+1+chunk_size]
```

Do not change `--action-offset 1` to zero unless the recorder is changed to save
pre-step observations.

## Install into the repository

From the AutoLabSim repository root, copy this add-on's `autolabsim/learning`
and `scripts/*.py` files into the matching folders, then install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-act.txt
```

## 1. Inspect data

```bash
python scripts/inspect_act_dataset.py data/episodes/adp_tip_to_tube_batch
```

Confirm that all episodes contain `action`, `ctrl`, and the selected image keys.

## 2. Generate enough image episodes

Example:

```bash
python scripts/generate_task_batch.py \
  --scene <your_scene> \
  --task adp_tip_to_tube \
  --count 100 \
  --with-images \
  --cameras overview_camera,wrist_cam,wrist_cam1 \
  --out-root data/episodes/adp_act_train
```

Use the exact scene/task command already verified in your repository when its
scene profile name differs.

## 3. Smoke-test training

```bash
python scripts/train_act.py \
  --data-root data/episodes/adp_act_train \
  --out-dir outputs/act_adp_smoke \
  --cameras overview_camera,wrist_cam,wrist_cam1 \
  --epochs 2 \
  --batch-size 4 \
  --num-workers 0 \
  --hidden-dim 128 \
  --nheads 4 \
  --encoder-layers 2 \
  --decoder-layers 2 \
  --dim-feedforward 512
```

## 4. Full training baseline

```bash
python scripts/train_act.py \
  --data-root data/episodes/adp_act_train \
  --out-dir outputs/act_adp \
  --cameras overview_camera,wrist_cam,wrist_cam1 \
  --state-key ctrl \
  --action-offset 1 \
  --chunk-size 50 \
  --batch-size 16 \
  --epochs 100 \
  --learning-rate 1e-4 \
  --kl-weight 10 \
  --num-workers 4
```

Checkpoints are written to `best.pt` and `last.pt`.

## 5. Generic deployment

```bash
python scripts/deploy_act.py \
  --checkpoint outputs/act_adp/best.pt \
  --scene <your_scene> \
  --seed 0 \
  --execute-steps 5 \
  --max-steps 1000 \
  --viewer \
  --out-dir data/act_rollouts/seed_0000
```

## Current task-mechanics limitation

The generic runner only sends the recorded actuator vector to `env.step`.
Current scripted AutoLabSim tasks also contain non-actuator transitions:

- ADP/pipette flows use kinematic `SiteAttachment` and fixed-joint state.
- Screw-cap execution maintains a separate commanded screw twist.

Those transitions are not fully represented by the current `action` array. A
robot trajectory can be learned, but a complete end-to-end rollout needs one of:

1. physically simulated grasp/thread mechanics, or
2. a task-specific deployment adapter that triggers the same transitions from
   geometric/contact conditions, or
3. regenerated data containing explicit auxiliary action/event channels.

For the first training check, use the generic runner to verify arm motion and
camera/state normalization. Add the task adapter before judging full task
success.
