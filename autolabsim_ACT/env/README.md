# AutoLabSim ACT Conda 环境

## 1. Ubuntu 系统依赖

```bash
sudo apt update
sudo apt install -y libgl1 libegl1 libglfw3 libglew2.2 libosmesa6 ffmpeg
```

## 2. 自动创建环境

把本目录复制到 AutoLabSim 仓库中，然后执行：

```bash
bash setup_act_env.sh auto
```

也可以显式指定 PyTorch CUDA wheel：

```bash
bash setup_act_env.sh cu118
bash setup_act_env.sh cu126
bash setup_act_env.sh cu128
bash setup_act_env.sh cpu
```

## 3. 激活

```bash
conda activate autolabsim-act
cd ~/user_lcy/AutoLabSim
```

## 4. 验证训练脚本

```bash
python scripts/train_act.py --help
python scripts/inspect_act_dataset.py --help
```

## 5. MuJoCo 渲染

有桌面窗口：

```bash
python scripts/deploy_act.py ... --viewer
```

无桌面、需要 GPU 离屏渲染：

```bash
MUJOCO_GL=egl python scripts/deploy_act.py ...
```

CPU 离屏渲染：

```bash
MUJOCO_GL=osmesa python scripts/deploy_act.py ...
```
