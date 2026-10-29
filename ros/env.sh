#!/bin/bash
# Source in EVERY terminal that talks to the TurtleBot 4:
#   source ros/env.sh
# (kept out of ~/.bashrc on purpose so it does not clash with other robots/domains)
_ROS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=23                       # leader TurtleBot 4 domain
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$_ROS_DIR/fastdds_pc.xml"
unset CYCLONEDDS_URI
unset ROS_DISCOVERY_SERVER
