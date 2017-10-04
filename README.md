![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


How to run the code?

1) ecelinux server

make clean

make


2) FPGA

cd sdscc_hw_1stconv_full  <dir> 

make


How to run the code?

1) ecelinux server

./darknet yolo test cfg/yolov1/tiny-yolo.cfg tiny-yolo.weights data/dog.jpg


2) FPGA

./darknet.elf yolo test cfg/yolov1/tiny-yolo.cfg tiny-yolo.weights data/dog.jpg

