FROM ros:noetic

ENV ROS_DISTRO=noetic

RUN sudo apt update
RUN apt-get install curl
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN sudo apt update
RUN apt-get install -y ros-${ROS_DISTRO}-desktop-full

RUN apt-get install -y ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-usb-cam

RUN mkdir -p /root/catkin_ws/src
COPY visual_servoing  /root/catkin_ws/src/visual_servoing

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && cd /root/catkin_ws && catkin_make && . devel/setup.sh

WORKDIR /root/catkin_ws

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
    && echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc
