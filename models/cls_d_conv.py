import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from conv_util_new import conv_module, conv_dir_module, conv_res_module, conv_decode_module


def placeholder_inputs(batch_size, num_point, use_normal=True):
    ''' npoint: num of points sampled in farthest point sampling '''
    if use_normal:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    else:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    direc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    cls_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    return pointclouds_pl, cls_pl, direc_pl


def get_model(point_cloud, direction, is_training, bn_decay=None, use_normal=True, k=2):
    '''
    :param point_cloud: [batch_size, ndataset, 3]
    :param direction: [bs, ndataset, 6], X and Y directions
    :param is_training:
    :param bn_decay:
    :param k:topk
    :return:
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    # end_points = {}
    l0_xyz = point_cloud[:, :, 0:3]
    if use_normal:
        l0_points = point_cloud[:, :, 3:6]
    else:
        l0_points = None
    # end_points['l0_points'] = l0_points

    with tf.variable_scope('encoder') as sc:
        l0_points = tf_util.conv1d(point_cloud, 32, 1, padding='VALID', bn=True, is_training=is_training, scope='fc0_0',
                                   bn_decay=bn_decay)
        l0_points = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc0_1',
                                   bn_decay=bn_decay)
        l1_points, l1_xyz, l1_dir = conv_dir_module(l0_xyz, direction, l0_points, npoint=512, radius=0.1, mlp=[64],
                                                    conv=[64], mlp2=[128], is_training=is_training,
                                                    bn_decay=bn_decay, scope='layer1', kernel_size=7, bn=True,
                                                    use_nchw=True, center=True)
        l2_points, l2_xyz, l2_dir = conv_res_module(l1_xyz, l1_dir, l1_points, radius=0.15, mlp=[64],
                                                    conv=[64], mlp2=[128], resnum=5, is_training=is_training,
                                                    bn_decay=bn_decay, scope='layer2', kernel_size=7, bn=True,
                                                    use_nchw=True, center=True)
        l3_points, l3_xyz, l3_dir = conv_dir_module(l2_xyz, l2_dir, l2_points, npoint=128, radius=0.2, mlp=[64],
                                                    conv=[64], mlp2=[128], is_training=is_training,
                                                    bn_decay=bn_decay, scope='layer3', kernel_size=7, bn=True,
                                                    use_nchw=True, center=True)
        l4_points, l4_xyz, l4_dir = conv_res_module(l3_xyz, l3_dir, l3_points, radius=0.2, mlp=[64],
                                                    conv=[64], mlp2=[128], resnum=5, is_training=is_training,
                                                    bn_decay=bn_decay, scope='layer4', kernel_size=7, bn=True,
                                                    use_nchw=True, center=True)
        l0_topk, l0_indices = tf.nn.top_k(tf.transpose(l0_points, [0, 2, 1]), k=k, name='l0_topk')
        l1_topk, l1_indices = tf.nn.top_k(tf.transpose(l1_points, [0, 2, 1]), k=k, name='l1_topk')
        l2_topk, l2_indices = tf.nn.top_k(tf.transpose(l2_points, [0, 2, 1]), k=k, name='l2_topk')
        l3_topk, l3_indices = tf.nn.top_k(tf.transpose(l3_points, [0, 2, 1]), k=k, name='l3_topk')
        l4_topk, l4_indices = tf.nn.top_k(tf.transpose(l4_points, [0, 2, 1]), k=k, name='l4_topk')
        l5_points = tf.concat([l0_topk, l1_topk, l2_topk, l3_topk, l4_topk], axis=1)

        net_cls = tf.reshape(l5_points, [batch_size, -1])
        net_cls = tf_util.fully_connected(net_cls, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net_cls = tf_util.dropout(net_cls, keep_prob=0.5, is_training=is_training, scope='dp1')
        net_cls = tf_util.fully_connected(net_cls, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net_cls = tf_util.dropout(net_cls, keep_prob=0.5, is_training=is_training, scope='dp2')
        net_cls = tf_util.fully_connected(net_cls, 40, activation_fn=None, scope='fc3')  # 40
    return net_cls


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


def get_cls_loss(pred, cls):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=cls)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('cls_classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 6))
        dir = tf.zeros((32, 1024, 6))
        output = get_model(inputs, dir, tf.constant(True))
        print(output)

