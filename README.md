# GAVEL-TD3

Minimal training release for GAVEL-TD3 mobile-robot navigation. The policy
combines a graph-attention state encoder with TD3 and trains in ROS1/Gazebo.
Only files required for training are included; checkpoints, logs, evaluation
results, figures, and pretrained weights are excluded.

## Repository layout

```text
algorithm/                  GAVEL-TD3, replay buffer, and Gazebo environment
catkin_ws/src/              ROS packages, robot model, world, and LiDAR plugin
gazebo_models/              custom models used by the training world
scripts/train.sh            build-and-train entry point
requirements.txt            tested Python dependencies
```

## 1. System dependencies

The tested stack is Ubuntu 20.04, ROS Noetic, Gazebo 11, Python 3.8, and a
CUDA-capable GPU. Install ROS Noetic first, then install the required packages:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake python3-pip python3-rosdep \
  ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control \
  ros-noetic-joint-state-publisher ros-noetic-robot-state-publisher \
  ros-noetic-xacro
```

The world uses Gazebo's `ActorCollisionsPlugin`. Build it once from the
official Gazebo Classic source:

```bash
git clone --depth 1 --branch gazebo11 \
  https://github.com/gazebosim/gazebo-classic.git /tmp/gazebo-classic
cmake -S /tmp/gazebo-classic/examples/plugins/actor_collisions \
  -B /tmp/gazebo-classic/examples/plugins/actor_collisions/build
cmake --build /tmp/gazebo-classic/examples/plugins/actor_collisions/build -j"$(nproc)"
sudo install -m 755 \
  /tmp/gazebo-classic/examples/plugins/actor_collisions/build/libActorCollisionsPlugin.so \
  /usr/lib/x86_64-linux-gnu/gazebo-11/plugins/
```

## 2. Clone and install Python packages

```bash
git clone https://github.com/liuzh678/GAVEL-TD3.git
cd GAVEL-TD3

# Install a CUDA build of PyTorch matching the local driver. This is the
# command used for the tested CUDA 12.1 environment.
python3 -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
```

For a different CUDA version, replace the PyTorch command with the matching
command from the official PyTorch installation selector.

## 3. Start training

```bash
bash scripts/train.sh
```

The launcher builds and sources the bundled catkin workspace, registers the
Gazebo models, checks the required actor-collision plugin, and starts training
with the verified defaults. Outputs are created under `outputs/` and ignored by
Git. To use another disk:

```bash
GAVEL_OUTPUT_DIR=/path/to/output bash scripts/train.sh
```

The experiment parameters near the end of `scripts/train.sh` can be overridden
with environment variables. For example:

```bash
RUN_ID=1 STEP_PENALTY=-0.0008 bash scripts/train.sh
```

## Monitor training

```bash
tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006
```

Then open <http://127.0.0.1:6006/>.
