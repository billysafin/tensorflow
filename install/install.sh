#!/bin/sh

#install python3
echo "Installing python3 with cpu support tensorflow."
sudo apt-get install python3-pip python3-dev python-virtualenv
virtualenv --system-site-packages -p python3
source ~/tensorflow/bin/activate

#install all the necessary libraries
echo "Installing all the neccessary libraries."
pip install --upgrade -r requirments.txt

echo "Complete installing and setting up the enviroment."