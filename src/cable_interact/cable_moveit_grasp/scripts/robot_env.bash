# Source this file in each Mini PC terminal; do not execute it as a child shell.
for cable_env_file in \
  /opt/ros/humble/setup.bash \
  /home/flexcycle/franka_ros2_ws/install/setup.bash \
  /home/flexcycle/Motion_Correlation/cmcor/pointcloud/config/fastdds/mini_pc_mtu1400.xml; do
  if [[ ! -r "$cable_env_file" ]]; then
    printf 'Required environment/config file is missing: %s\n' "$cable_env_file" >&2
    unset cable_env_file
    return 1
  fi
done
unset cable_env_file
source /opt/ros/humble/setup.bash || return
source /home/flexcycle/franka_ros2_ws/install/setup.bash || return
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/flexcycle/Motion_Correlation/cmcor/pointcloud/config/fastdds/mini_pc_mtu1400.xml
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
# Load the manual dual-arm commands into this terminal.
source "${BASH_SOURCE[0]%/*}/dual_arm_commands.bash" || return
