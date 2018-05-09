TensorFlow
====
Overview

## Description
This package is for image recongization and others

## Requirement
Python3
Ubuntu16.4
TensorFlow12

##to change parameters
1. add tensorflow way of adding paramters after the command, e.g. python main.py --parameters
2. open setting.py under each folder, except web will be helper.py, and chage the parameters.

## Usage
1. to gather images
  A. run image_downloader.py under gather/images/flicker
  B. run conver_to_tfrecords.py under gather/images
  C. run main.py under learning/image/learn
  D. cd ../
  E. run conver_to_tfrecords.py

2. to gather text (under development)
  A. run twitter.py under gather/text/twitter
  B. ...

3. to study
  * remeber to have images saved to saved_dir/data/images/
  A. cd saved_dir/learning/image/learn
  B. python main.py

4. to use tensorboard
  A. ssh -L 8888:localhost:9999 user@xxx.xxx.xxx.xxx
  B. tensorboard --logdir=saved_dir/log --port 9999

5. to use flask + TensorFlow
  * REMEBER you need to have studied checkpoints and/or protocal buff files.
  A. ssh -L 5000:localhost:5000 user@xxx.xxx.xxx.xxx
  B. cd saved_dir/web
  C. python web.py
  D. open your browser and enter http://127.0.0.1:5000

## Install
1. With AVX2 FMA
  Please refer to "How to install.txt": with AVX2/FMA
2. Without
    Please refer to "How to install.txt": without
    or run install/install.sh

## Reference
Please refer to Reference.txt
