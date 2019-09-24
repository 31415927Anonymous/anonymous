### Installation
The code is tested under TF1.9.0 GPU version and Python 3.6.8

Some basic operation, like farthest point sampling, are form implementation of PointNet++.

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

### Usage

We used the pre-processed data of Pointnet++. To get the direction dataset, run the code in `code_for_directions`.

#### Shape Classification

You can get the sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6GB)</a>. Move the uncompressed data folder to `data/modelnet40_normal_resampled`

And you should move the direction dataset to `data/modelnet40_normal_resampled/patch_mat/directions`

ModelNet40:

        python train_di_cnn.py --normal
        python eval_di_cnn.py --num_votes 12 --normal

#### Object Part Segmentation

Preprocessed ShapeNetPart dataset (XYZ, normal and part labels) can be found <a href="https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip">here (674MB)</a>. Move the uncompressed data folder to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

Then you need to move the direction dataset to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/directions_seg`

ShapeNet:

        python train_di_seg.py
        python eval_di_seg.py --repeat_num 24

#### S3DIS

S3DIS needs to be pre-processed by partitioning blocks, which is form implementation of Pointcnn. Run the code in `S3DIS`.