import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, select_top_k
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import math


def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    # print((b, n, c, m))
    # print((xyz1, (b,1,n,c)))
    xyz1 = tf.expand_dims(xyz1, 1)
    xyz2 = tf.expand_dims(xyz2, 2)
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)
    # print((dist, k))
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])
    # print((idx, val))
    # val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx


def get_weight(xyz, nbhd, radius):
    '''
    :param xyz: [B,np,3]
    :param nbhd: [B,np,nb,3]
    :param radius: float
    :return:
    '''
    xyz = tf.expand_dims(xyz, 2)
    dist = tf.reduce_sum((xyz - nbhd) ** 2, -1)  # [B,np,nb]
    weight = tf.maximum(1-tf.sqrt(dist)/radius, 0.0)
    return tf.expand_dims(weight, 2)


def nb_knn_point(k, nbhd, nbhd_idx, xyz):
    '''
    :param k: int32, number of k in k-nn search
    :param nbhd: [batch_size, npoint, nbhd_num, c], neighborhood of xyz
    :param nbhd_idx: [batch_size, npoint, nbhd_num], index of nbhd in dataset
    :param xyz: [batch_size, npoint, c] float32 array, query points
    :return: idx: [batch_size, npoint, k] int32 array, indices to input points
    '''
    b = tf.shape(xyz)[0]
    n = tf.shape(xyz)[1]
    c = tf.shape(xyz)[2]
    nbn = tf.shape(nbhd)[2]
    xyz = tf.tile(tf.expand_dims(xyz, 2), [1, 1, nbn, 1])
    dist = tf.reduce_sum((xyz - nbhd) ** 2, -1)  # [b, n, nbn]
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])  # [batch_size, npoint, k]
    idx2 = tf.expand_dims(idx, 3)
    idx0 = tf.tile(tf.reshape(tf.range(b), [-1, 1, 1, 1]), [1, n, k, 1])
    idx1 = tf.tile(tf.reshape(tf.range(n), [1, n, 1, 1]), [b, 1, k, 1])
    idx = tf.gather_nd(nbhd_idx, tf.concat([idx0, idx1, idx2], axis=3))  # [b, n, k, 3]
    return idx


def conv_module(xyz, new_xyz, points, weight, nbhd_idx, mlp, conv, mlp2, is_training, bn_decay, scope, bn=True, use_xyz=True,
                use_nchw=False, mlp2ac=False, center=False, use_pooling=False):
    '''
    :param xyz: [B,nd,3]
    :param new_xyz: [B,np,3]
    :param points: [B,nd,c]
    :param weight: [B,np,ks,nb]
    :param nbhd_idx: [B,np,nb]
    :return:
    '''
    ### mlp2 must be 1 layer !!!!
    data_format = 'NCHW' if use_nchw else 'NHWC'
    ker_ac = tf.nn.relu
    if mlp2ac:
        ac = tf.nn.relu
    elif mlp2 != None:
        ac = None
    else:
        ker_ac = None
    with tf.variable_scope(scope) as sc:
        npoint = weight.get_shape()[1].value
        # print(patch.get_shape())
        kernel_size = weight.get_shape()[2].value
        batch_size = tf.shape(weight)[0]
        if points is None:
            points = xyz
        elif use_xyz:
            points = tf.concat([xyz, points], axis=2)
        # channel = tf.shape(points)[2]
        channel = points.get_shape()[2].value

        new_points = group_point(points, nbhd_idx)  # (batch_size, npoint, nbhd_num, 3+channel)

        if center:
            center_xyz = tf.expand_dims(new_xyz, 2)
            new_points = tf.concat([new_points[:, :, :, 0:3] - center_xyz, new_points[:, :, :, 3:]], axis=-1)

        # pooling
        if use_pooling:
            pooling = tf.reduce_max(new_points, axis=[2], name='pooling')
            # print(pooling.get_shape())
            if mlp2 != None:
                endchannel = mlp2[-1]
                ac = None
            else:
                endchannel = conv[-1]
                ker_ac = None
            if channel != endchannel:
                pooling = tf_util.conv1d(pooling, endchannel, 1, padding='VALID', activation_fn=None,
                                         scope='pooling_conv')

        if use_nchw:
            new_points = tf.transpose(new_points, [0, 3, 1, 2])
        if mlp is not None:
            for i, num_out_channel in enumerate(mlp):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_1_%d' % i, bn_decay=bn_decay,
                                            data_format=data_format)
        if use_nchw:
            new_points = tf.transpose(new_points, [0, 2, 3, 1])
        new_points = tf.matmul(weight, new_points)  # [B,np,ks,nb]*[B,np,nb,c]  -->  [B,np,ks,c]
        if use_nchw:
            new_points = tf.transpose(new_points, [0, 3, 1, 2])

        new_points = tf_util.conv2d(new_points, num_output_channels=conv[0], kernel_size=[1, kernel_size],
                                    padding='VALID', stride=[1, 1], bn=bn, is_training=is_training, scope='ker_conv',
                                    bn_decay=bn_decay, data_format=data_format, activation_fn=ker_ac)

        if mlp2 is not None:
            new_points = tf_util.conv2d(new_points, mlp2[0], [1, 1], padding='VALID', stride=[1, 1], bn=bn,
                                        is_training=is_training, scope='conv_2', bn_decay=bn_decay,
                                        data_format=data_format, activation_fn=ac)

        if use_nchw:
            new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])
        if use_pooling:
            return new_points + pooling
        else:
            return new_points


def get_patch(xyz0, xyz, xyz_x, xyz_y, kernel_size, radius, nbhd_num=8):
    '''
    :param xyz0: [Batch_size, ndataset, 3], float
    :param xyz: [Batch_size, npoint, 3], float
    :param xyz_x: [Batch_size, npoint, 3], float
    :param xyz_y: [Batch_size, npoint, 3], float
    :param kernel_size: int
    :param radius: float
    :param k: int, use ave or max of k points instead of 1
    :return: patch: [Batch_size, npoint, kernel_size], int
    '''
    _, nbhd_idx = knn_point(nbhd_num, xyz0, xyz)
    nbhd = group_point(xyz0, nbhd_idx)
    k = int(((math.sqrt(kernel_size)) - 1) / 2)
    weight = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            xyz_ij = xyz + radius * i * xyz_x + radius * j * xyz_y
            w = get_weight(xyz_ij, nbhd, radius)
            weight.append(w)
    weight = tf.concat(weight, axis=2)  # [B,np,ks,nb]
    return weight, nbhd_idx


def get_patch_rot(xyz0, xyz, xyz_x, xyz_y, kernel_size, radius=0.3, nbhd_num=12):
    '''
    :param xyz0: [Batch_size, ndataset, 3], float
    :param xyz: [Batch_size, npoint, 3], float
    :param xyz_x: [Batch_size, npoint, 3], float
    :param xyz_y: [Batch_size, npoint, 3], float
    :param kernel_size: int
    :param radius: float
    :param k: int, use ave or max of k points instead of 1
    :param nbhd: int,
    :return: patch: [Batch_size, npoint, kernel_size], int
    '''
    _, nbhd_idx = knn_point(nbhd_num, xyz0, xyz)    # [B,np,nb]
    nbhd = group_point(xyz0, nbhd_idx)   # [B,np,nb,3]
    k = kernel_size - 1
    if kernel_size == 13:
        k = int(k / 2)
        nbhd_num = 16
    elif kernel_size == 19:
        k = int(k * 1.0 / 3)
        nbhd_num *= 3
    weight = []
    w = get_weight(xyz, nbhd, radius)  # weight [B,np,1,nb]
    weight.append(w)
    for i in range(k):
        a = np.cos(2 * np.pi * i / 6)
        b = np.sin(2 * np.pi * i / 6)
        xyz_ij = xyz + radius * a * xyz_x + radius * b * xyz_y
        w = get_weight(xyz_ij, nbhd, radius)
        weight.append(w)
    if kernel_size >= 13:
        radius *= math.sqrt(3)
        for i in range(k):
            a = np.cos(2 * np.pi * (i + 0.5) / 6)
            b = np.sin(2 * np.pi * (i + 0.5) / 6)
            xyz_ij = xyz + radius * a * xyz_x + radius * b * xyz_y
            w = get_weight(xyz_ij, nbhd, radius)
            weight.append(w)
    if kernel_size == 19:
        radius *= 2
        for i in range(k):
            a = np.cos(2 * np.pi * i / 6)
            b = np.sin(2 * np.pi * i / 6)
            xyz_ij = xyz + radius * a * xyz_x + radius * b * xyz_y
            w = get_weight(xyz_ij, nbhd, radius)
            weight.append(w)
    weight = tf.concat(weight, axis=2)   # [B,np,ks,nb]
    return weight, nbhd_idx


def conv_dir_module(xyz, direction, points, npoint, radius, mlp, conv, mlp2, is_training, bn_decay, scope,
                    kernel_size=9, bn=True, use_xyz=True, use_nchw=False, center=False):
    with tf.variable_scope(scope) as sc:
        ids = farthest_point_sample(npoint, xyz)
        new_xyz = gather_point(xyz, ids)
        x_dir = gather_point(direction[:, :, 0:3], ids)
        y_dir = gather_point(direction[:, :, 3:6], ids)
        new_direction = tf.concat([x_dir, y_dir], axis=2)
        if kernel_size in [7, 13, 19]:
            weight, nbhd_idx = get_patch_rot(xyz, new_xyz, x_dir, y_dir, kernel_size, radius)
        else:
            weight, nbhd_idx = get_patch(xyz, new_xyz, x_dir, y_dir, kernel_size, radius)
        new_points = conv_module(xyz, new_xyz, points, weight, nbhd_idx, mlp=mlp, conv=conv, mlp2=mlp2,
                                 is_training=is_training, bn_decay=bn_decay, center=center,
                                 scope=scope, bn=bn, use_xyz=use_xyz, use_nchw=use_nchw, use_pooling=True)
        new_points = tf.nn.relu(new_points)
        return new_points, new_xyz, new_direction


def conv_res_module(xyz, direction, points, radius, mlp, conv, mlp2, resnum, is_training, bn_decay, scope,
                    kernel_size=9, bn=True, use_xyz=True, use_nchw=False, all=False, center=False):
    with tf.variable_scope(scope) as sc:
        new_xyz = xyz
        x_dir = direction[:, :, 0:3]
        y_dir = direction[:, :, 3:6]
        new_direction = direction
        if kernel_size in [7, 13, 19]:
            weight, nbhd_idx = get_patch_rot(xyz, new_xyz, x_dir, y_dir, kernel_size, radius)
        else:
            weight, nbhd_idx = get_patch(xyz, new_xyz, x_dir, y_dir, kernel_size, radius)
        pset = []
        for i in range(resnum):
            # print(points)
            new_points = conv_module(xyz, new_xyz, points, weight, nbhd_idx, mlp=mlp, conv=conv, mlp2=mlp2,
                                     is_training=is_training, bn_decay=bn_decay, scope='res_%d' % i, bn=bn,
                                     use_xyz=use_xyz, use_nchw=use_nchw, center=center, mlp2ac=False)
            # print(new_points)
            points = new_points + points
            points = tf.nn.relu(points)
            pset.append(points)
        if all:
            points = tf.concat(pset, axis=2)
        return points, new_xyz, new_direction


def conv_decode_module(xyz, new_xyz, new_direction, new_point, points, radius, mlp, conv, mlp2, is_training, bn_decay,
                       scope, kernel_size=7, bn=True, use_xyz=True, use_nchw=False, use_new=True, center=False):
    with tf.variable_scope(scope) as sc:
        x_dir = new_direction[:, :, 0:3]
        y_dir = new_direction[:, :, 3:6]
        if kernel_size in [7, 13, 19]:
            weight, nbhd_idx = get_patch_rot(xyz, new_xyz, x_dir, y_dir, kernel_size, radius)
        else:
            weight, nbhd_idx = get_patch(xyz, new_xyz, x_dir, y_dir, kernel_size, radius)
        new_points = conv_module(xyz, new_xyz, points, weight, nbhd_idx, mlp=mlp, conv=conv, mlp2=mlp2,
                                 is_training=is_training, bn_decay=bn_decay, center=center,
                                 scope=scope, bn=bn, use_xyz=use_xyz, use_nchw=use_nchw, use_pooling=True)
        new_points = tf.nn.relu(new_points)
        if use_new:
            new_points = tf.concat([new_point, new_points], axis=2)
        return new_points


if __name__ == '__main__':
    ks = 9
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    inp = []
    inp1 = []
    for i in range(8):
        for j in range(8):
            inp.append(i * x + j * y)
            if (i + j) % 2 == 0:
                inp1.append(i * x + j * y)
    inp = np.stack(inp)
    inp1 = np.stack(inp1)
    with tf.Graph().as_default():
        sess = tf.Session()
        inputs = tf.constant(inp, dtype=tf.float32)
        inputs = tf.reshape(inputs, [1, 64, 3])
        inputs1 = tf.constant(inp1, dtype=tf.float32)
        inputs1 = tf.reshape(inputs1, [1, 32, 3])
        xx = tf.constant(x, dtype=tf.float32)
        xn = tf.reshape(xx, [1, 1, 3])
        xn1 = tf.tile(xn, [1, 32, 1])
        yy = tf.constant(y, dtype=tf.float32)
        yn = tf.reshape(yy, [1, 1, 3])
        yn1 = tf.tile(yn, [1, 32, 1])
        out = get_patch(inputs, inputs1, xn1, yn1, ks)
        output = sess.run(out)
        print(output)
