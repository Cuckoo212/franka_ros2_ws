# Six manual commands. Each call blocks until the requested operation completes.
# The dual_arm launch must be running with execution_enabled:=true.
# Clear the former misspelled functions when reloading an existing terminal.
unset -f grap_a grap_b
grasp_a() { ros2 service call /dual_arm/dual_arm_motion/grasp_a std_srvs/srv/Trigger '{}'; }
grasp_b() { ros2 service call /dual_arm/dual_arm_motion/grasp_b std_srvs/srv/Trigger '{}'; }
fix_a() { ros2 service call /dual_arm/dual_arm_motion/fix_a std_srvs/srv/Trigger '{}'; }
fix_b() { ros2 service call /dual_arm/dual_arm_motion/fix_b std_srvs/srv/Trigger '{}'; }
home_a() { ros2 service call /dual_arm/dual_arm_motion/home_a std_srvs/srv/Trigger '{}'; }
home_b() { ros2 service call /dual_arm/dual_arm_motion/home_b std_srvs/srv/Trigger '{}'; }
# Automatic coordinator: idle until explicitly started. Pause does not cancel an in-flight motion.
alternate_start() { ros2 service call /cable_moveit_alternate_grasp/start std_srvs/srv/Trigger '{}'; }
alternate_pause() { ros2 service call /cable_moveit_alternate_grasp/pause std_srvs/srv/Trigger '{}'; }
alternate_resume() { ros2 service call /cable_moveit_alternate_grasp/resume std_srvs/srv/Trigger '{}'; }
alternate_reset() { ros2 service call /cable_moveit_alternate_grasp/reset std_srvs/srv/Trigger '{}'; }
alternate_state() { ros2 topic echo /cable_moveit_alternate_grasp/state --once --qos-durability transient_local --field data; }
