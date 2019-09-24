import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import shapenet_direction_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='seg_d_conv', help='Model name [default: seg_d_conv]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=121, help='Epoch to run [default: 121]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.002]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=150000, help='Decay step for lr decay [default: 150000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--direction', default='directions_seg', help='directions')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
WEIGHT_DECAY = 1e-8

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
os.system('cp train_di_seg.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 50

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TRAIN_DATASET = shapenet_direction_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False,
                                                             split='trainval', dir_root=FLAGS.direction,
                                                             batch_size=BATCH_SIZE, return_cls_label=True)
TEST_DATASET = shapenet_direction_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False,
                                                            split='test', dir_root=FLAGS.direction,
                                                            batch_size=BATCH_SIZE, return_cls_label=True)


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
            pointclouds_pl, labels_pl, direc_pl, cls_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred = MODEL.get_model(pointclouds_pl, direc_pl, cls_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_focal_loss(pred, labels_pl)
            reglosses = tf.get_collection('reglosses')
            total_loss = WEIGHT_DECAY * tf.add_n(reglosses, name='total_loss') + loss
            tf.summary.scalar('total_loss', total_loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
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
                boundaries = [500000, 800000, 1100000, 1600000, 2000000]
                lr_sgd = [0.1, 0.01, 0.003, 0.001, 0.0003, 0.0001]
                step = batch * BATCH_SIZE
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
        ckpt = tf.train.get_checkpoint_state('log')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'dir_pl': direc_pl,
               'cls_pl': cls_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            # eval_one_epoch(sess, ops, test_writer, testp_writer)
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer, testp_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps, normal, seg = dataset[idxs[i + start_idx]]
        batch_data[i, :, 0:3] = ps
        batch_data[i, :, 3:6] = normal
        batch_label[i, :] = seg
    return batch_data, batch_label


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
    cur_batch_dir = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    cur_batch_cls = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_normal, batch_label, batch_dir, batch_cls = TRAIN_DATASET.next_batch(augment=True)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = np.concatenate([batch_data, batch_normal], axis=2)
        cur_batch_label[0:bsize, ...] = batch_label
        cur_batch_dir[0:bsize, ...] = batch_dir
        cur_batch_cls[0:bsize] = batch_cls

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['dir_pl']: cur_batch_dir,
                     ops['labels_pl']: cur_batch_label,
                     ops['cls_pl']: cur_batch_cls,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
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
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    is_training = False

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    seg_classes = TEST_DATASET.seg_classes
    shape_ious = {cat: [] for cat in list(seg_classes.keys())}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in list(seg_classes.keys()):
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
    cur_batch_dir = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    cur_batch_cls = np.zeros((BATCH_SIZE), dtype=np.int32)

    while TEST_DATASET.has_next_batch():
        batch_data, batch_normal, batch_label, batch_dir, batch_cls = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = np.concatenate([batch_data, batch_normal], axis=2)
        cur_batch_label[0:bsize, ...] = batch_label
        cur_batch_dir[0:bsize, ...] = batch_dir
        cur_batch_cls[0:bsize] = batch_cls

        # ---------------------------------------------------------------------
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['dir_pl']: cur_batch_dir,
                     ops['labels_pl']: cur_batch_label,
                     ops['cls_pl']: cur_batch_cls,
                     ops['is_training_pl']: is_training, }
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        # ---------------------------------------------------------------------

        # Select valid data
        cur_pred_val = pred_val[0:bsize]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((bsize, NUM_POINT)).astype(np.int32)
        for i in range(bsize):
            cat = seg_label_to_cat[cur_batch_label[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
        correct = np.sum(cur_pred_val == cur_batch_label)
        total_correct += correct
        total_seen += (bsize * NUM_POINT)
        loss_sum += loss_val
        c_batch_label = cur_batch_label[0:bsize, :]
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(c_batch_label == l)
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (c_batch_label == l)))

        for i in range(bsize):
            segp = cur_pred_val[i, :]
            segl = cur_batch_label[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    for cat in list(shape_ious.keys()):
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_shape_ious = np.mean(list(shape_ious.values()))
    log_string('eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET) / BATCH_SIZE)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f' % (cat, shape_ious[cat]))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))

    dev_summary = tf.Summary()
    dev_summary.value.add(tag="loss", simple_value=loss_sum / float(len(TEST_DATASET) / BATCH_SIZE))
    dev_summary.value.add(tag="accuracy", simple_value=total_correct / float(total_seen))
    dev_summary.value.add(tag="avg class acc", simple_value=np.mean(
        np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)))
    dev_summary.value.add(tag="mean mIoU", simple_value=mean_shape_ious)
    dev_summary.value.add(tag="mean mIoU(all shapes)", simple_value=np.mean(all_shape_ious))
    testp_writer.add_summary(dev_summary, step)

    EPOCH_CNT += 1
    TEST_DATASET.reset()
    return total_correct / float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
