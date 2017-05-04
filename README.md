# dispflownet-tf

This is Tensorflow implementation of DispNet from https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/MIFDB16.pdf

Currently DispNetCorr1D is implemented and converges in experiments.

Model | FlyingThings3D EPE | FPS (K40)
-------|-----|-------
DispNetCorr1D (slow) | ~3 | 5

## Correlation Layer

There are two versions of custom Correlation1D layer: one is built upon simple tf operations like pad and slice in "for" loop - that is the slow one. Faster implementation utilizes the CUDA code from original Caffe layer and allows to inference the model in exactly the same time as in Caffe, but this implementation is not fully tested yet.
