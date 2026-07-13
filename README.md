# AutoLabSim

AutoLabSim is a MuJoCo-based simulation project for laboratory manipulation tasks. The current codebase focuses on cap manipulation, pipette manipulation, and episode generation/export workflows.

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

- `autolabsim/tasks/screw_cap.py`
  Dual-arm cap unscrewing task.
- `autolabsim/tasks/pipette_grasp.py`
  Pipette grasping and tip-mounting task.
- `autolabsim/task_target.py`
  Shared task target, frame reference, and gripper command definitions.
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

- `scripts/generate_task_batch.py`
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
python scripts/view_scene.py model/scenes/scene_mujoco_fast_tubes_pipette.xml \
  --reset-config configs/reset_pipette_tips_random_subset.json
```
python scripts/view_scene.py model/scenes/scene_mujoco_fast_tubes_pipette_grasped.xml \
 --reset-config configs/reset_pipette_grasped.json
### Generate Episodes with Images

```bash
python3 scripts/generate_screw_cap_batch.py \
  --count 1 \
  --with-images \
  --cameras overview_camera,wrist_cam,wrist_cam1 \
  --out-root data/episodes/screw_cap_batch
```
```bash
python3 scripts/generate_pipette_grasp_batch.py \
  --count 1 \
  --with-images \
  --cameras overview_camera,wrist_cam,wrist_cam1 \
  --out-root data/episodes/pipette_grasp_batch
```

### Export Episode Video

```bash
python3 scripts/export_episode_images.py \
  data/episodes/screw_cap_batch \
  --camera overview_camera,wrist_cam,wrist_cam1 \
  --format mp4 \
  --fps 20
```

```bash
python3 scripts/export_episode_images.py \
  data/episodes/pipette_grasp_batch \
  --camera overview_camera,wrist_cam,wrist_cam1 \
  --format mp4 \
  --fps 20
```
# GitHub Upload, Authentication and Team Collaboration

本项目 GitHub 仓库信息：

```text
Repository: AutoLabSim
Owner: bumingli2000-stack
Default branch: main
HTTPS remote: https://github.com/bumingli2000-stack/AutoLabSim.git
SSH remote: git@github.com:bumingli2000-stack/AutoLabSim.git
```

本文分为两部分：

1. 仓库管理员如何邀请新成员；
2. 新成员如何从零配置 GitHub、下载代码并参与开发。

---

# 一、仓库管理员：邀请新成员

仅有仓库地址并不代表新人拥有推送权限。

对于个人账号名下的仓库，管理员需要先邀请对方成为 collaborator。对方接受邀请后，才具备仓库读写权限。GitHub 个人仓库的 collaborator 可以拉取代码并推送修改。

操作步骤：

1. 打开 GitHub 上的 `AutoLabSim` 仓库；
2. 点击仓库顶部的 `Settings`；
3. 在左侧进入 `Collaborators`；
4. 点击 `Add people`；
5. 输入新成员的 GitHub 用户名或邮箱；
6. 发送邀请；
7. 通知新成员登录 GitHub 并接受邀请。

建议直接使用对方的 GitHub 用户名邀请，避免邮箱不匹配。

新成员必须接受邀请后才能直接向仓库推送代码。

---

# 二、新成员：注册并接受邀请

## 1. 创建 GitHub 账号

新人首先需要注册自己的 GitHub 账号，并完成邮箱验证。

账号创建完成后，把自己的 GitHub 用户名发给仓库管理员。

## 2. 接受仓库邀请

管理员发送邀请后，新成员应：

1. 登录自己的 GitHub；
2. 查看 GitHub 通知或注册邮箱；
3. 打开仓库协作邀请；
4. 点击接受邀请。

接受邀请后，再继续配置本地 Git。

---

# 三、安装 Git 和 SSH 工具

以下命令适用于 Ubuntu：

```bash
sudo apt update
sudo apt install -y git openssh-client
```

检查安装：

```bash
git --version
ssh -V
```

---

# 四、配置本地 Git 身份

Git 提交记录中的姓名和邮箱来自本地 Git 配置，不是由 SSH 密钥自动决定的。

设置姓名：

```bash
git config --global user.name "你的姓名或GitHub用户名"
```

设置邮箱：

```bash
git config --global user.email "你的GitHub邮箱"
```

例如：

```bash
git config --global user.name "Zhang San"
git config --global user.email "zhangsan@example.com"
```

查看配置：

```bash
git config --global --list
```

也可以分别检查：

```bash
git config --global user.name
git config --global user.email
```

建议使用已经添加并验证到 GitHub 账号中的邮箱，否则提交记录可能无法正确关联到自己的 GitHub 账号。Git 的 `user.name` 和 `user.email` 只影响后续提交，不会修改过去的提交记录。

---

# 五、推荐认证方式：SSH 密钥

## 1. SSH 密钥的基本概念

SSH 密钥包含两个文件：

```text
私钥：id_ed25519
公钥：id_ed25519.pub
```

必须遵守：

* 私钥只能保存在自己的电脑中；
* 私钥不能上传到 GitHub；
* 私钥不能发给其他人；
* 私钥不能提交到 Git 仓库；
* 只把 `.pub` 结尾的公钥添加到 GitHub。

每位开发者都应使用自己的 GitHub 账号和自己的 SSH 密钥，不要多人共用同一个密钥。

---

## 2. 检查电脑上是否已有 SSH 密钥

执行：

```bash
ls -al ~/.ssh
```

重点查看是否存在：

```text
id_ed25519
id_ed25519.pub
```

或者：

```text
id_rsa
id_rsa.pub
```

GitHub 建议在生成新密钥前先检查本机是否已有可用密钥。

如果已经有用于本人 GitHub 账号的 SSH 密钥，可以继续使用，不必重复生成。

如果没有，按照下一步生成。

---

## 3. 生成新的 SSH 密钥

将下面的邮箱替换为自己的 GitHub 邮箱：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

例如：

```bash
ssh-keygen -t ed25519 -C "zhangsan@example.com"
```

终端会询问保存位置：

```text
Enter file in which to save the key
```

直接按 Enter，默认保存为：

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

随后会询问是否设置 passphrase。

建议设置一个自己能记住的密码，以提高私钥安全性。GitHub 当前推荐优先生成 Ed25519 密钥；旧系统不支持 Ed25519 时才使用 RSA 4096。

旧系统可以使用：

```bash
ssh-keygen -t rsa -b 4096 -C "你的GitHub邮箱"
```

---

## 4. 启动 ssh-agent

执行：

```bash
eval "$(ssh-agent -s)"
```

将私钥加入 ssh-agent：

```bash
ssh-add ~/.ssh/id_ed25519
```

检查已加载的密钥：

```bash
ssh-add -l
```

如果私钥设置了 passphrase，首次添加时需要输入该密码。`ssh-agent` 可以代为管理密钥，避免每次 Git 操作都重新输入。

---

## 5. 查看并复制 SSH 公钥

执行：

```bash
cat ~/.ssh/id_ed25519.pub
```

输出内容类似：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... zhangsan@example.com
```

复制完整的一整行。

注意：复制的是：

```text
id_ed25519.pub
```

绝对不要复制或上传：

```text
id_ed25519
```

---

## 6. 把公钥添加到 GitHub

登录自己的 GitHub 账号，然后依次进入：

```text
头像
→ Settings
→ SSH and GPG keys
→ New SSH key
```

填写：

```text
Title: 当前电脑的名称，例如 Ubuntu-Lab-PC
Key type: Authentication Key
Key: 粘贴 id_ed25519.pub 的完整内容
```

最后点击：

```text
Add SSH key
```

添加的是用户自己的 GitHub 账号，不是仓库管理员的账号。

---

## 7. 测试 SSH 连接

执行：

```bash
ssh -T git@github.com
```

第一次连接时可能出现：

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入：

```text
yes
```

成功时会出现类似：

```text
Hi YOUR_USERNAME! You've successfully authenticated, but GitHub does not provide shell access.
```

出现这条消息说明 SSH 认证成功。

这里显示的用户名必须是新成员自己的 GitHub 用户名。

---

# 六、新成员首次下载项目

注意：

> 仓库已经存在时，新成员不应该重新执行 `git init`、`Initial commit` 或重新创建远程仓库。

应直接克隆已有仓库。

进入希望存放项目的目录：

```bash
mkdir -p ~/projects
cd ~/projects
```

使用 SSH 克隆：

```bash
git clone git@github.com:bumingli2000-stack/AutoLabSim.git
```

进入项目：

```bash
cd AutoLabSim
```

检查远程仓库：

```bash
git remote -v
```

正常结果类似：

```text
origin  git@github.com:bumingli2000-stack/AutoLabSim.git (fetch)
origin  git@github.com:bumingli2000-stack/AutoLabSim.git (push)
```

检查当前分支：

```bash
git branch --show-current
```

应显示：

```text
main
```

获取最新代码：

```bash
git pull --rebase origin main
```

---

# 七、已有本地项目时切换到 SSH

如果项目原来已经通过 HTTPS 克隆，不需要重新下载。

进入项目：

```bash
cd /home/你的用户名/projects/AutoLabSim
```

检查当前远程：

```bash
git remote -v
```

将远程地址修改为 SSH：

```bash
git remote set-url origin git@github.com:bumingli2000-stack/AutoLabSim.git
```

再次检查：

```bash
git remote -v
```

测试拉取：

```bash
git fetch origin
```

如果没有权限错误，说明配置成功。

---

# 八、推荐的日常开发流程

多人协作时，不建议所有人都长期直接向 `main` 推送。

推荐把 `main` 作为稳定分支，新功能使用独立分支。

## 1. 开始工作前同步 main

进入项目：

```bash
cd ~/projects/AutoLabSim
```

切换到 main：

```bash
git switch main
```

拉取最新代码：

```bash
git pull --rebase origin main
```

检查状态：

```bash
git status
```

---

## 2. 创建功能分支

例如开发 LeRobot 数据导出功能：

```bash
git switch -c feature/export-lerobot-data
```

修复移液枪规划问题：

```bash
git switch -c fix/pipette-planner
```

增加新任务：

```bash
git switch -c feature/screw-cap-task
```

推荐分支命名：

```text
feature/功能名称
fix/问题名称
refactor/重构内容
docs/文档内容
test/测试内容
```

---

## 3. 修改并检查文件

查看修改：

```bash
git status
```

查看具体差异：

```bash
git diff
```

查看准备提交的差异：

```bash
git diff --staged
```

---

## 4. 提交修改

优先添加本次相关文件，不要无条件把所有临时文件都提交：

```bash
git add path/to/file1 path/to/file2
```

确认暂存内容：

```bash
git status
git diff --staged
```

提交：

```bash
git commit -m "Refactor pipette target planning"
```

提交信息示例：

```bash
git commit -m "Add screw cap reset config"
git commit -m "Fix tube grasp point generation"
git commit -m "Refactor pipette attachment planning"
git commit -m "Add LeRobot episode exporter"
```

提交应尽量满足：

* 一次提交只处理一个明确问题；
* 不混入无关格式修改；
* 提交信息说明实际改了什么；
* 不使用大量含义不明的提交信息，例如 `update`、`fix`、`test123`。

---

## 5. 推送功能分支

第一次推送当前分支：

```bash
git push -u origin HEAD
```

或者明确写分支名：

```bash
git push -u origin feature/export-lerobot-data
```

后续继续推送：

```bash
git push
```

然后在 GitHub 上创建 Pull Request，请其他成员检查后合并到 `main`。

---

# 九、直接更新 main 的流程

只有在项目规模较小、修改已经确认，并且团队允许直接推送 `main` 时使用。

```bash
cd ~/projects/AutoLabSim
git switch main
git pull --rebase origin main
git status
git add <修改的文件>
git commit -m "Describe your changes"
git push origin main
```

不要在没有同步远程代码时直接推送。

不要对 `main` 使用：

```bash
git push --force
```

除非仓库管理员明确知道并批准该操作。

---

# 十、功能分支如何同步最新 main

开发过程中，如果远程 `main` 有了新提交：

```bash
git fetch origin
git rebase origin/main
```

如果当前分支尚未推送，可以正常继续：

```bash
git push
```

如果当前分支已经推送过，而 rebase 改写了该功能分支的历史，可以使用：

```bash
git push --force-with-lease
```

只能对自己的功能分支使用：

```bash
git push --force-with-lease
```

不要对共享的 `main` 分支使用。

`--force-with-lease` 比 `--force` 更安全，因为远程分支出现未预期的新提交时，它会拒绝覆盖。

---

# 十一、处理 rebase 冲突

执行：

```bash
git pull --rebase origin main
```

或：

```bash
git rebase origin/main
```

如果出现冲突，先检查：

```bash
git status
```

冲突文件中会出现：

```text
<<<<<<< HEAD
当前一方内容
=======
另一方内容
>>>>>>> commit
```

人工决定最终保留内容，并删除这些冲突标记。

解决一个文件后：

```bash
git add <已经解决的文件>
```

继续 rebase：

```bash
git rebase --continue
```

如果还有冲突，重复：

```bash
git status
git add <已经解决的文件>
git rebase --continue
```

取消整个 rebase：

```bash
git rebase --abort
```

不要在没有理解冲突内容的情况下直接全部选择某一方。

---

# 十二、查看提交历史

查看最近提交：

```bash
git log --oneline --graph --decorate -n 15
```

查看所有本地和远程分支：

```bash
git branch -a
```

查看某次提交内容：

```bash
git show <commit-id>
```

查看某个文件的历史：

```bash
git log --oneline -- path/to/file
```

---

# 十三、恢复和撤销修改

## 1. 丢弃尚未暂存的单个文件修改

```bash
git restore path/to/file
```

## 2. 取消暂存，但保留本地修改

```bash
git restore --staged path/to/file
```

## 3. 撤销一个已经提交并推送的提交

团队仓库优先使用：

```bash
git revert <commit-id>
```

这会创建一个新的反向提交，不会破坏已有历史。

不要轻易使用：

```bash
git reset --hard
```

该命令可能永久丢失未提交内容。

---

# 十四、HTTPS 认证方式

SSH 是本项目推荐方式。

如果由于网络或环境限制必须使用 HTTPS，可以使用：

```bash
git clone https://github.com/bumingli2000-stack/AutoLabSim.git
```

但 GitHub 已经取消了命令行 Git 操作中的账号密码认证。使用 HTTPS 推送时，密码输入框中不能填写 GitHub 登录密码，而要填写 Personal Access Token，或者使用 GitHub CLI/Git Credential Manager。

## 1. 创建 Personal Access Token

在 GitHub 中依次进入：

```text
头像
→ Settings
→ Developer settings
→ Personal access tokens
→ Fine-grained tokens
→ Generate new token
```

建议：

```text
Token name: AutoLabSim-Ubuntu
Expiration: 设置合理的有效期
Repository access: 仅选择 AutoLabSim
Repository permissions:
    Contents: Read and write
```

Fine-grained token 可以限定到特定资源所有者、特定仓库和特定权限，比授予宽泛权限更安全。

生成后应立即保存 token，因为页面关闭后通常不能再次查看完整 token。

禁止：

* 把 token 写进代码；
* 把 token 写进 README；
* 把 token 提交到 Git；
* 把 token 发给其他开发者；
* 多人共用同一个 token。

每个人应创建和管理自己的认证凭据。

## 2. HTTPS 推送时输入方式

执行：

```bash
git push
```

提示用户名时，输入自己的 GitHub 用户名。

提示密码时，粘贴 Personal Access Token，而不是 GitHub 登录密码。

---

# 十五、GitHub CLI 认证方式

HTTPS 用户也可以安装 GitHub CLI，通过浏览器完成认证。

安装：

```bash
sudo apt update
sudo apt install -y gh
```

登录：

```bash
gh auth login
```

通常依次选择：

```text
GitHub.com
HTTPS
Login with a web browser
```

检查认证状态：

```bash
gh auth status
```

GitHub 官方推荐 HTTPS 用户使用 GitHub CLI 或 Git Credential Manager 管理凭据，避免手动反复输入 token。

---

# 十六、Clash 代理配置

如果终端无法直接访问 GitHub，并且本机 Clash HTTP 代理端口为 `7890`，可以临时设置：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

检查：

```bash
echo "$http_proxy"
echo "$https_proxy"
```

随后执行：

```bash
git fetch origin
git pull
git push
```

取消临时代理：

```bash
unset http_proxy
unset https_proxy
```

也可以只为当前 Git 命令使用：

```bash
https_proxy=http://127.0.0.1:7890 git pull
```

注意：

```text
http_proxy 和 https_proxy 主要影响 HTTPS 连接，
不一定会自动代理普通 SSH 端口 22。
```

如果 HTTPS 可以访问，但：

```bash
ssh -T git@github.com
```

一直超时，可以选择：

1. 使用 HTTPS + PAT/GitHub CLI；
2. 让 Clash 接管系统 SSH 流量；
3. 使用 GitHub SSH over port 443。

---

# 十七、SSH 端口 22 被阻止时使用端口 443

先测试：

```bash
ssh -T -p 443 git@ssh.github.com
```

如果成功，编辑：

```bash
nano ~/.ssh/config
```

加入：

```text
Host github.com
    HostName ssh.github.com
    User git
    Port 443
    IdentityFile ~/.ssh/id_ed25519
```

设置配置文件权限：

```bash
chmod 600 ~/.ssh/config
```

再次测试：

```bash
ssh -T git@github.com
```

GitHub 官方提供通过 `ssh.github.com:443` 使用 SSH 的方案，适用于普通 SSH 端口被网络阻止的环境。

---

# 十八、常见错误排查

## 1. `Permission denied (publickey)`

检查 SSH 是否认证成功：

```bash
ssh -T git@github.com
```

检查密钥是否加载：

```bash
ssh-add -l
```

如果没有密钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

检查 GitHub 中添加的是公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

---

## 2. `Repository not found`

常见原因：

* 仓库地址写错；
* 没有接受 collaborator 邀请；
* 当前 SSH 密钥属于另一个 GitHub 账号；
* 当前 GitHub 账号没有仓库权限。

检查当前 SSH 对应账号：

```bash
ssh -T git@github.com
```

检查远程地址：

```bash
git remote -v
```

SSH 地址应为：

```text
git@github.com:bumingli2000-stack/AutoLabSim.git
```

---

## 3. `Authentication failed`

如果远程地址是 HTTPS：

```bash
git remote -v
```

GitHub 登录密码不能用于 Git 推送。

应选择：

* Personal Access Token；
* GitHub CLI；
* Git Credential Manager；
* 改用 SSH。

---

## 4. `rejected non-fast-forward`

说明远程分支包含本地没有的提交。

先同步：

```bash
git pull --rebase origin main
```

解决可能出现的冲突后再推送：

```bash
git push origin main
```

不要直接使用：

```bash
git push --force
```

---

## 5. 提示没有 Git 用户名或邮箱

如果出现：

```text
Author identity unknown
```

执行：

```bash
git config --global user.name "你的姓名"
git config --global user.email "你的GitHub邮箱"
```

---

## 6. 查看当前到底使用 SSH 还是 HTTPS

执行：

```bash
git remote get-url origin
```

如果结果以：

```text
git@github.com:
```

开头，使用的是 SSH。

如果结果以：

```text
https://github.com/
```

开头，使用的是 HTTPS。

---

# 十九、禁止提交的内容

本仓库已经忽略：

```text
.vscode/
data/
MUJOCO_LOG.TXT
```

所有开发者还应避免提交：

```text
SSH 私钥
Personal Access Token
密码
API Key
大型数据集
模型权重
临时日志
Python 缓存
虚拟环境
本地 IDE 配置
```

提交前必须检查：

```bash
git status
git diff --staged
```

检查某个文件是否被 `.gitignore` 忽略：

```bash
git check-ignore -v path/to/file
```

常见建议忽略项：

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.env
*.log
.vscode/
data/
outputs/
checkpoints/
MUJOCO_LOG.TXT
```

不要把真实 token 或密码直接写入 `.env` 示例文件。

可以提交：

```text
.env.example
```

但里面只能放占位符：

```text
GITHUB_TOKEN=your_token_here
```

不能放真实凭据。

---

# 二十、仓库管理员首次上传

下面流程只适用于仓库管理员第一次创建并上传仓库。

如果远程仓库已经有代码，新成员不要执行这一节。

```bash
cd /home/buming/projects/AutoLabSim
git status
git add .
git commit -m "Initial commit"
git push -u origin main
```

如果尚未添加远程：

使用 SSH：

```bash
git remote add origin git@github.com:bumingli2000-stack/AutoLabSim.git
```

或者使用 HTTPS：

```bash
git remote add origin https://github.com/bumingli2000-stack/AutoLabSim.git
```

确认：

```bash
git remote -v
```

首次推送：

```bash
git push -u origin main
```

---

# 二十一、新成员最简操作清单

完成一次性配置：

```bash
sudo apt update
sudo apt install -y git openssh-client

git config --global user.name "你的姓名"
git config --global user.email "你的GitHub邮箱"

ssh-keygen -t ed25519 -C "你的GitHub邮箱"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

把输出的公钥添加到自己的 GitHub：

```text
Settings
→ SSH and GPG keys
→ New SSH key
```

测试：

```bash
ssh -T git@github.com
```

克隆：

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:bumingli2000-stack/AutoLabSim.git
cd AutoLabSim
```

开始新功能：

```bash
git switch main
git pull --rebase origin main
git switch -c feature/my-feature
```

提交并推送：

```bash
git status
git add <修改的文件>
git commit -m "Describe the change"
git push -u origin HEAD
```

最后在 GitHub 创建 Pull Request。

---

# 二十二、安全原则

1. 每个成员使用自己的 GitHub 账号；
2. 每个成员使用自己的 SSH 密钥或 token；
3. 私钥只保存在本机；
4. GitHub 只添加公钥；
5. token 和私钥都不能发送给其他人；
6. 不要提交密码、密钥和 token；
7. 不要直接强制推送 `main`；
8. 开始工作前先同步远程；
9. 非简单修改优先使用功能分支；
10. 合并前检查代码和提交内容。
