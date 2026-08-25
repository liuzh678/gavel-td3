#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
PYTHON="${PYTHON:-python3}"

[[ -f "$ROS_SETUP" ]] || {
  echo "ROS setup not found: $ROS_SETUP" >&2
  exit 2
}

set +u
source "$ROS_SETUP"
set -u

if [[ ! -f "$REPO_ROOT/catkin_ws/devel/setup.bash" ]]; then
  (
    cd "$REPO_ROOT/catkin_ws"
    catkin_make
  )
fi
set +u
source "$REPO_ROOT/catkin_ws/devel/setup.bash"
set -u

export GAZEBO_MODEL_PATH="$REPO_ROOT/gazebo_models:${GAZEBO_MODEL_PATH:-}"
export GAVEL_OUTPUT_DIR="${GAVEL_OUTPUT_DIR:-$REPO_ROOT/outputs}"
export PYTHONUNBUFFERED=1

plugin_name="libActorCollisionsPlugin.so"
plugin_found=""
IFS=':' read -r -a gazebo_plugin_dirs <<< "${GAZEBO_PLUGIN_PATH:-}"
gazebo_plugin_dirs+=(
  "/usr/lib/x86_64-linux-gnu/gazebo-11/plugins"
  "/usr/local/lib"
)
for plugin_dir in "${gazebo_plugin_dirs[@]}"; do
  if [[ -n "$plugin_dir" && -f "$plugin_dir/$plugin_name" ]]; then
    plugin_found="$plugin_dir/$plugin_name"
    break
  fi
done
if [[ -z "$plugin_found" ]]; then
  echo "$plugin_name was not found. Build it using the README instructions." >&2
  exit 2
fi

SCALE_TO_GOAL="${SCALE_TO_GOAL:-0.0}"
OBS_SCALE="${OBS_SCALE:-40}"
LOG_TEST="${LOG_TEST:-100}"
RUN_ID="${RUN_ID:-333}"
UP_SPEED="${UP_SPEED:-740}"
SMOOTHNESS_SCALE="${SMOOTHNESS_SCALE:-0}"
ACCELERATION_SCALE="${ACCELERATION_SCALE:-0}"
OU_NOISE_DECAY_STEPS="${OU_NOISE_DECAY_STEPS:-500}"
STEP_PENALTY="${STEP_PENALTY:--0.0008}"
ALPHA_L="${ALPHA_L:-46}"

mkdir -p "$GAVEL_OUTPUT_DIR"
cd "$REPO_ROOT/algorithm"

exec "$PYTHON" train_gavel_td3.py \
  "$SCALE_TO_GOAL" \
  "$OBS_SCALE" \
  "$LOG_TEST" \
  "$RUN_ID" \
  "$UP_SPEED" \
  "$SMOOTHNESS_SCALE" \
  "$ACCELERATION_SCALE" \
  "$OU_NOISE_DECAY_STEPS" \
  "$STEP_PENALTY" \
  "$ALPHA_L"
