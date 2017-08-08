TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
nvcc -std=c++11 -c -o shift_corr.cu.o shift_corr.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o shift_corr.so shift_corr.cc shift_corr.cu.o -I $TF_INC -fPIC -lcudart -L/usr/local/cuda/lib64 #-D_GLIBCXX_USE_CXX11_ABI=0
