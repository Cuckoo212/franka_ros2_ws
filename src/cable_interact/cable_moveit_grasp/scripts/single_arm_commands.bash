# Source robot_env.bash first. Override CABLE_SINGLE_ARM_TASK_NAMESPACE if needed.
single_preview() { ros2 service call "/${CABLE_SINGLE_ARM_TASK_NAMESPACE:-single_arm}/single_arm_motion/preview_selected" std_srvs/srv/Trigger '{}'; }
single_execute() { ros2 service call "/${CABLE_SINGLE_ARM_TASK_NAMESPACE:-single_arm}/single_arm_motion/execute_selected" std_srvs/srv/Trigger '{}'; }
single_release() { ros2 service call "/${CABLE_SINGLE_ARM_TASK_NAMESPACE:-single_arm}/single_arm_motion/release" std_srvs/srv/Trigger '{}'; }
single_reset() { ros2 service call "/${CABLE_SINGLE_ARM_TASK_NAMESPACE:-single_arm}/single_arm_motion/reset" std_srvs/srv/Trigger '{}'; }
single_state() { ros2 topic echo "/${CABLE_SINGLE_ARM_TASK_NAMESPACE:-single_arm}/single_arm_motion/status" --once --qos-durability transient_local --field data; }
