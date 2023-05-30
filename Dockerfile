FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-usb-cam

ENV WS=/root/catkin_ws
RUN mkdir -p ${WS}/src
WORKDIR ${WS}
COPY control  src/control
COPY perception src/perception
COPY wam_common src/wam_common

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && catkin_make && . devel/setup.sh

COPY requirements.txt requirements.txt
RUN apt-get install -y python3-pip git
RUN apt-get install -y python-dateutil
RUN python3 -m pip install -r requirements.txt

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
    && echo "source $WS/devel/setup.bash" >> ~/.bashrc

RUN sed --in-place --expression '$isource "$WS/devel/setup.bash"' /ros_entrypoint.sh
