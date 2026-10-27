#!/bin/bash
set -e
cable_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$cable_script_dir/robot_env.bash"
exec /usr/bin/python3 "$cable_script_dir/start_robot.py"
