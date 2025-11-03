FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-c"]

# Install system dependencies and Python packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gdb git wget autoconf automake nano vim tmux curl \
    python3-dev python3-pip python3-scipy python3-matplotlib

RUN pip3 install --break-system-packages jupyter fiftyone requests unidecode tiktoken orjson

WORKDIR /colcon_ws/src/vlm_gist

# Copy requirements file and install Python dependencies without upgrading pip
COPY requirements.txt /colcon_ws/src/vlm_gist/requirements.txt
RUN pip3 install --break-system-packages -r requirements.txt

# Copy project files
COPY . /colcon_ws/src/vlm_gist

# Clone ros2 dependencies and build workspace
WORKDIR /colcon_ws/src
RUN git clone https://github.com/AIS-Bonn/nimbro_utils.git
RUN git clone https://github.com/AIS-Bonn/nimbro_api_interfaces.git
RUN git clone https://github.com/AIS-Bonn/nimbro_api.git

WORKDIR /colcon_ws
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /colcon_ws/install/local_setup.bash" >> /root/.bashrc

# Entrypoint to source ROS and workspace environments
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source /colcon_ws/install/setup.bash && exec bash"]
