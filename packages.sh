pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip cache purge
pip install opencv-python-headless==4.5.5.64

#!/bin/bash
echo "deb http://deb.debian.org/debian bullseye main contrib non-free" > /etc/apt/sources.list.d/opencv.list
echo "deb http://deb.debian.org/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list.d/opencv.list
apt-get update
apt-get install -y libopencv-dev
