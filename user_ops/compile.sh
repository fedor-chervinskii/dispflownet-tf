nvcc -std=c++11 -c -o shift_corr.cu.o shift_corr.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o shift_corr.so shift_corr.cc shift_corr.cu.o -I $TF_INC -fPIC -lcudart
