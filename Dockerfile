FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-usb-cam

RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws
COPY control  src/control
COPY perception src/perception
COPY wam_common src/wam_common

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && catkin_make && . devel/setup.sh

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
    && echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

# COPY ./entrypoint.sh /
# ENTRYPOINT ["/entrypoint.sh"]