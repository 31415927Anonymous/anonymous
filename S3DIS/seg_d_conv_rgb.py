import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from conv_util import conv_module, conv_dir_module, conv_res_module, conv_decode_module


def placeholder_inputs(batch_size, num_point):
    ''' npoint: num of points sampled in farthest point sampling '''
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    direc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    return pointclouds_pl, labels_pl, direc_pl


def get_model(point_cloud, direction, is_training, bn_decay=None, k=2):
    '''
    :param point_cloud: [batch_size, ndataset, 3]
    :param direction: [bs, ndataset, 6], X and Y directions
    :param is_training:
    :param bn_decay:
    :param k:topk
    :return:
    '''
    center = False
    i = 2
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud[:, :, 0:3]

    # encoder
    l0_points = tf_util.conv1d(tf.concat([point_cloud, direction], axis=-1), 32, 1, padding='VALID', bn=True, is_training=is_training, scope='fc0_0',
                               bn_decay=bn_decay)
    l0_points = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc0_1',
                               bn_decay=bn_decay)
    l1_points, l1_xyz, l1_dir = conv_dir_module(l0_xyz, direction, l0_points, npoint=512, radius=0.15, mlp=[64*i*2],
                                                conv=[64*i*2], mlp2=[128*i*2], is_training=is_training,
                                                bn_decay=bn_decay, scope='layer1', kernel_size=13, bn=True,
                                                use_nchw=True, center=center)
    l2_points, l2_xyz, l2_dir = conv_res_module(l1_xyz, l1_dir, l1_points, radius=0.2, mlp=[64*i*2],
                                                conv=[64*i*2], mlp2=[128*i*2], resnum=5, is_training=is_training,
                                                bn_decay=bn_decay, scope='layer2', kernel_size=7, bn=True,
                                                use_nchw=True, center=center)
    l3_points, l3_xyz, l3_dir = conv_dir_module(l2_xyz, l2_dir, l2_points, npoint=128, radius=0.25, mlp=[64*i],
                                                conv=[64*i], mlp2=[128*i], is_training=is_training,
                                                bn_decay=bn_decay, scope='layer3', kernel_size=7, bn=True,
                                                use_nchw=True, center=center)
    l4_points, l4_xyz, l4_dir = conv_res_module(l3_xyz, l3_dir, l3_points, radius=0.3, mlp=[64*i],
                                                conv=[64*i], mlp2=[128*i], resnum=15, is_training=is_training,
                                                bn_decay=bn_decay, scope='layer4', kernel_size=7, bn=True,
                                                use_nchw=True, center=center)
    l0_topk, l0_indices = tf.nn.top_k(tf.transpose(l0_points, [0, 2, 1]), k=k, name='l0_topk')
    l1_topk, l1_indices = tf.nn.top_k(tf.transpose(l1_points, [0, 2, 1]), k=k, name='l1_topk')
    l2_topk, l2_indices = tf.nn.top_k(tf.transpose(l2_points, [0, 2, 1]), k=k, name='l2_topk')
    l3_topk, l3_indices = tf.nn.top_k(tf.transpose(l3_points, [0, 2, 1]), k=k, name='l3_topk')
    l4_topk, l4_indices = tf.nn.top_k(tf.transpose(l4_points, [0, 2, 1]), k=k, name='l4_topk')
    l5_points = tf.concat([l1_topk, l2_topk, l3_topk, l4_topk], axis=1)
    l5channel = l5_points.get_shape()[1] * k

    # decoder
    np = l4_points.get_shape()[1].value
    l4_points_d = tf.tile(tf.reshape(l5_points, [-1, 1, l5channel]), [1, np, 1])
    l4_points_d = tf.concat([l4_points_d, l4_points, l3_points], axis=2)
    l4_points_d = conv_decode_module(l4_xyz, l3_xyz, l3_dir, l3_points, l4_points_d, radius=0.2, mlp=[512*i, 256*i], conv=[128*i],
                                     mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='dlayer4', use_new=False,
                                     kernel_size=7, bn=True, use_xyz=True, use_nchw=False, center=center)
    l3_points_d, _, _ = conv_res_module(l3_xyz, l3_dir, l4_points_d, radius=0.2, mlp=[64*i],
                                        conv=[64*i], mlp2=[128*i], resnum=5, is_training=is_training,
                                        bn_decay=bn_decay, scope='dlayer3', kernel_size=7, bn=True,
                                        use_nchw=True, center=center)
    l2_points_d = pointnet_fp_module(l2_xyz, l3_xyz, tf.concat([l1_points, l2_points], axis=-1), l3_points_d, [], is_training, bn_decay,
                                     scope='fp_layer2')
    l2_points_d = conv_decode_module(l2_xyz, l1_xyz, l1_dir, l1_points, l2_points_d, radius=0.1, mlp=None, conv=[128*i],
                                     mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='dlayer2', use_new=False,
                                     kernel_size=7, bn=True, use_xyz=True, use_nchw=False, center=center)
    l1_points_d, _, _ = conv_res_module(l1_xyz, l1_dir, l2_points_d, radius=0.1, mlp=[64*i],
                                        conv=[64*i], mlp2=[128*i], resnum=5, is_training=is_training,
                                        bn_decay=bn_decay, scope='dlayer1', kernel_size=7, bn=True,
                                        use_nchw=True, center=center)
    l0_points_d = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_d, [], is_training, bn_decay,
                                     scope='fp_layer1')
    # FC layers
    net = tf_util.conv1d(l0_points_d, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc0',
                         bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp0')
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1',
                         bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 13, 1, padding='VALID', activation_fn=None, scope='fc2')
    return net


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


def focal_loss(cls_weight, gamma=2, balance=False):
    def FL(pred, label):
        from tensorflow.python.ops import array_ops

        pred = tf.nn.softmax(pred, name='softmax')
        zeros = array_ops.zeros_like(pred, dtype=pred.dtype)    # BxNx50
        label_one_hot = tf.one_hot(label, depth=13, on_value=1.0, off_value=0.0)   # BxNx50
        one_minus_p = array_ops.where(tf.greater(label_one_hot, zeros), label_one_hot - pred, zeros)
        FL_loss = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0))

        if balance:
            classes_weight = array_ops.zeros_like(pred, dtype=pred.dtype)
            classes_w_tensor = tf.convert_to_tensor(cls_weight, dtype=pred.dtype)
            classes_weight += classes_w_tensor
            alpha = array_ops.where(tf.greater(label_one_hot, zeros), classes_weight, zeros)
            FL_loss = alpha * FL_loss

        FL_loss = tf.reduce_sum(FL_loss, axis=-1)
        return FL_loss
    return FL


def get_focal_loss(pred, label, weight=[1.0]*13):
    """ pred: BxNxC,
        label: BxN, """
    loss = focal_loss(weight)(pred=pred, label=label)
    classify_loss = 2.5*tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 6))
        dir = tf.zeros((32, 1024, 6))
        output = get_model(inputs, dir, tf.constant(True))
        print(output)
