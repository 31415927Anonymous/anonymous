#/bin/bash
/opt/software/cuda/9.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /opt/software/anaconda/3/lib/python3.6/site-packages/tensorflow/include -I /opt/software/cuda/9.0/include -I /opt/software/anaconda/3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /opt/software/cuda/9.0/lib64/ -L /opt/software/anaconda/3/lib/python3.6/site-packages/tensorflow/ -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
