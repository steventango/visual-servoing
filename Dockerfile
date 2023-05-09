FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

RUN sudo apt update
RUN apt-get install curl
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN sudo apt update
RUN apt-get install -y ros-${ROS_DISTRO}-desktop-full

RUN apt-get install -y ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-usb-cam

RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws
COPY control  src/control
COPY perception src/perception
COPY wam_common src/wam_common

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && catkin_make && . devel/setup.sh

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
    && echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]