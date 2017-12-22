# dispflownet-tf

Reimplementation in tensorflow of Dispnet and Dispnet-Corr1D.
Original code thx to [@fedor-chervinskii](https://github.com/fedor-chervinskii/dispflownet-tf)

## Improvement and fixes
+ Python3 conversion
+ Add compile.sh inside user_ops
+ Modified image preprocessing to match those used in caffe
+ Added implementation of ["Unsupervised Adaptation for Deep Stereo" - ICCV2017](https://github.com/CVLAB-Unibo/Unsupervised-Adaptation-for-Deep-Stereo)
+ Lots of arguments from command line
+ Native tensorflow input pipeline
+ Add improved code for inference

## Pretrained nets
+ Weights for Dispnet with tensorflow correlation layaer trained for 1200000 step on FlyingThings3D available [here](https://drive.google.com/open?id=1OHkH4rjHJIpd1fA_fRIFEo5q6Je79rUs) 
