# robot-demo

## Setup
```
# on host: 
# install nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo apt install libvulkan-dev vulkan-tools libgles2-mesa-dev -y  
xhost +

# Ensure the following flags are set on container: 
# --env NVIDIA_DRIVER_CAPABILITIES=all --rm --runtime=nvidia --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" 
export TARGET=[name of container]
docker exec -u root $TARGET bash -c "apt update && apt install libvulkan-dev vulkan-tools libglvnd-dev -y"
docker cp /usr/share/vulkan/icd.d/nvidia_icd.json $TARGET:/usr/share/vulkan/icd.d/nvidia_icd.json
docker cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json $TARGET:/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# for container
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```
