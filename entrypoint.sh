#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/noetic/setup.bash"
source /root/catkin_ws/devel/setup.bash
exec "$@"