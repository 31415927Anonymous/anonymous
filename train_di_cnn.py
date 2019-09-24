import numpy as np
import tensorflow as tf
import argparse
import math
from datetime import datetime
import h5py
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import modelnet_direction_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='cls_d_conv', help='Model name [default: cls_d_conv]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='sgd', help='adam or momentum or sgd [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--direction', default='directions', help='directions')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
WEIGHT_DECAY = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train_di_cnn.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

PATCH_SIZE = [512, 128]
KERNEL_SIZE = 9
NUM_CLASSES = 40

# Shapenet official train/test split
assert (NUM_POINT <= 10000)
DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
PATCH_PATH = os.path.join(DATA_PATH, 'patch_mat', FLAGS.direction)
TRAIN_DATASET = modelnet_direction_dataset.ModelNetDataset(root=DATA_PATH, patch_root=PATCH_PATH,
                                                           npoints=NUM_POINT, split='train',
                                                           normal_channel=FLAGS.normal, modelnet10=False,
                                                           batch_size=BATCH_SIZE)
TEST_DATASET = modelnet_direction_dataset.ModelNetDataset(root=DATA_PATH, patch_root=PATCH_PATH,
                                                          npoints=NUM_POINT, split='test',
                                                          normal_channel=FLAGS.normal, modelnet10=False,
                                                          batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, direc_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,
                                                                                       FLAGS.normal)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred = MODEL.get_model(pointclouds_pl, direc_pl, is_training_pl, bn_decay=bn_decay, use_normal=FLAGS.normal)
            loss = MODEL.get_loss(pred, labels_pl)
            reglosses = tf.get_collection('reglosses')
            total_loss = WEIGHT_DECAY * tf.add_n(reglosses, name='reg_loss') + loss
            tf.summary.scalar('total_loss', total_loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator

            if OPTIMIZER == 'momentum':
                learning_rate = get_learning_rate(batch)
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                learning_rate = get_learning_rate(batch)
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif OPTIMIZER == 'sgd':
                boundaries = [400000.1, 800000.1, 1200000.1, 1600000.1, 2000000.1, 2400000.1, 2800000.1, 3200000.1]
                lr_sgd = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0003, 0.0001]
                step = batch*BATCH_SIZE
                learning_rate = tf.train.piecewise_constant(step, boundaries=boundaries, values=lr_sgd)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            tf.summary.scalar('learning_rate', learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        testp_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test_plus'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'dir_pl': direc_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer, testp_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    is_training = True

    log_string(str(datetime.now()))

    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE,), dtype=np.int32)
    cur_batch_dir = np.zeros((BATCH_SIZE, NUM_POINT, 6))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label, batch_dir = TRAIN_DATASET.next_batch(augment=True)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        cur_batch_dir[0:bsize, ...] = batch_dir

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['dir_pl']: cur_batch_dir,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        if (batch_idx + 1) % 50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / 50))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()


def eval_one_epoch(sess, ops, test_writer, testp_writer):
    global EPOCH_CNT
    is_training = False
    num_votes = 4

    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_dir = np.zeros((BATCH_SIZE, NUM_POINT, 6))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label, batch_dir = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]

        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        cur_batch_dir[0:bsize, ...] = batch_dir

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES))
        for vote_idx in range(num_votes):
            shuffled_indices = np.arange(NUM_POINT)
            np.random.shuffle(shuffled_indices)
            if FLAGS.normal:
                rotated_data, rotated_dir = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],
                                                                                             cur_batch_dir[:, shuffled_indices, :],
                                                                                             vote_idx / float(num_votes) * np.pi * 2)
            else:
                rotated_data, rotated_dir = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                                                                                 cur_batch_dir[:, shuffled_indices, :],
                                                                                 vote_idx / float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['dir_pl']: rotated_dir,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            batch_pred_sum += pred_val
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    dev_summary = tf.Summary()
    dev_summary.value.add(tag="loss", simple_value=loss_sum / float(batch_idx))
    dev_summary.value.add(tag="accuracy", simple_value=total_correct / float(total_seen))
    dev_summary.value.add(tag="avg class acc", simple_value=np.mean(
        np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)))
    testp_writer.add_summary(dev_summary, step)

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct / float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
