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
import tf_util
import shapenet_direction_dataset
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='seg_d_conv', help='Model name [default: seg_d_conv]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log_eval]')
parser.add_argument('--num_point', default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--direction', default='directions_seg', help='directions')
parser.add_argument('--repeat_num', type=int, default=12, help='repeat num [default: 12]')
FLAGS = parser.parse_args()

REPEAT_NUM = FLAGS.repeat_num

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
NUM_CLASSES = 50

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TEST_DATASET = shapenet_direction_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT*REPEAT_NUM,
                                                            classification=False, split='test',
                                                            dir_root=FLAGS.direction, batch_size=BATCH_SIZE,
                                                            return_cls_label=True, state='eval')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, direc_pl, cls_pl = MODEL.placeholder_inputs(REPEAT_NUM, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            print("--- Get model and loss")
            pred = MODEL.get_model(pointclouds_pl, direc_pl, cls_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'dir_pl': direc_pl,
               'cls_pl': cls_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}

        eval_one_epoch(sess, ops)


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


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    num_batches = len(TEST_DATASET)

    total_correct_max = 0
    total_correct_ave = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    seg_classes = TEST_DATASET.seg_classes
    shape_ious_max = {cat: [] for cat in list(seg_classes.keys())}
    shape_ious_ave = {cat: [] for cat in list(seg_classes.keys())}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in list(seg_classes.keys()):
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    cur_batch_data = np.zeros((REPEAT_NUM, NUM_POINT, 6))
    cur_batch_label = np.zeros((REPEAT_NUM, NUM_POINT), dtype=np.int32)
    cur_batch_dir = np.zeros((REPEAT_NUM, NUM_POINT, 6))
    cur_batch_cls = np.zeros((REPEAT_NUM,), dtype=np.int32)

    for batch_idx in range(len(TEST_DATASET)):
        if batch_idx % 200 == 0:
            log_string('%03d/%03d' % (batch_idx, num_batches))
        data, normal, label, dire, cls, point_num = TEST_DATASET[batch_idx]
        batch_data = np.reshape(data, [REPEAT_NUM, NUM_POINT, 3])
        batch_normal = np.reshape(normal, [REPEAT_NUM, NUM_POINT, 3])
        batch_label = np.reshape(label, [REPEAT_NUM, NUM_POINT])
        batch_dir = np.reshape(dire, [REPEAT_NUM, NUM_POINT, 6])
        batch_cls = np.tile(cls, [REPEAT_NUM])

        idx = np.arange(NUM_POINT)
        np.random.shuffle(idx)
        idx_re = np.argsort(idx)

        bsize = REPEAT_NUM
        cur_batch_data[0:bsize, ...] = np.concatenate([batch_data, batch_normal], axis=-1)[:, idx, :]
        cur_batch_label[0:bsize, ...] = batch_label[:, idx]
        cur_batch_dir[0:bsize, ...] = batch_dir[:, idx, :]
        cur_batch_cls[0:bsize] = batch_cls

        # ---------------------------------------------------------------------
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['dir_pl']: cur_batch_dir,
                     ops['labels_pl']: cur_batch_label,
                     ops['cls_pl']: cur_batch_cls,
                     ops['is_training_pl']: is_training, }
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = pred_val[:, idx_re, :]
        # ---------------------------------------------------------------------

        # Select valid data

        cur_pred_logits = np.reshape(pred_val[0:bsize], (REPEAT_NUM*NUM_POINT, NUM_CLASSES))
        max_pred_val = np.zeros([point_num])
        ave_pred_val = np.zeros([point_num])
        for i in range(point_num):
            cat = seg_label_to_cat[label[0]]
            point_pred_logits = cur_pred_logits[i::point_num, seg_classes[cat]]
            ave_pred_val[i] = np.argmax(np.mean(point_pred_logits, axis=0)) + seg_classes[cat][0]
            max_pred_val[i] = (np.argmax(point_pred_logits) % len(seg_classes[cat])) + seg_classes[cat][0]
        cur_label = label[0:point_num]
        correct_max = np.sum(max_pred_val == cur_label)
        correct_ave = np.sum(ave_pred_val == cur_label)
        total_correct_max += correct_max
        total_correct_ave += correct_ave
        total_seen += point_num
        loss_sum += loss_val

        cur_pred_val = max_pred_val

        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_label == l)
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (cur_label == l)))
        segp_max = max_pred_val
        segp_ave = ave_pred_val
        segl = cur_label
        cat = seg_label_to_cat[segl[0]]
        part_ious_max = [0.0 for _ in range(len(seg_classes[cat]))]
        part_ious_ave = [0.0 for _ in range(len(seg_classes[cat]))]
        for l in seg_classes[cat]:
            if (np.sum(segl == l) == 0) and (np.sum(segp_max == l) == 0):  # part is not present, no prediction as well
                part_ious_max[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious_max[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp_max == l)) / float(np.sum((segl == l) | (segp_max == l)))
            if (np.sum(segl == l) == 0) and (np.sum(segp_ave == l) == 0):  # part is not present, no prediction as well
                part_ious_ave[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious_ave[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp_ave == l)) / float(np.sum((segl == l) | (segp_ave == l)))
        shape_ious_max[cat].append(np.mean(part_ious_max))
        shape_ious_ave[cat].append(np.mean(part_ious_ave))
        batch_idx += 1

    all_shape_ious_max = []
    all_shape_ious_ave = []
    for cat in list(shape_ious_max.keys()):
        for iou in shape_ious_max[cat]:
            all_shape_ious_max.append(iou)
        shape_ious_max[cat] = np.mean(shape_ious_max[cat])
    print(len(all_shape_ious_max))
    for cat in list(shape_ious_ave.keys()):
        for iou in shape_ious_ave[cat]:
            all_shape_ious_ave.append(iou)
        shape_ious_ave[cat] = np.mean(shape_ious_ave[cat])
    print(len(all_shape_ious_ave))
    mean_shape_ious_max = np.mean(list(shape_ious_max.values()))
    mean_shape_ious_ave = np.mean(list(shape_ious_ave.values()))

    log_string('eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET))))
    log_string('eval accuracy(max): %f' % (total_correct_max / float(total_seen)))
    log_string('eval accuracy(average): %f' % (total_correct_ave / float(total_seen)))
    log_string('eval avg class acc: %f' % (
    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    for cat in sorted(shape_ious_max.keys()):
        log_string('eval mIoU(max) of %s:\t %f' % (cat, shape_ious_max[cat]))
    log_string('eval mean mIoU(max): %f' % (mean_shape_ious_max))
    log_string('eval mean mIoU(max) (all shapes): %f' % (np.mean(all_shape_ious_max)))
    log_string('--------------------------------------------')
    for cat in sorted(shape_ious_ave.keys()):
        log_string('eval mIoU(ave) of %s:\t %f' % (cat, shape_ious_ave[cat]))
    log_string('eval mean mIoU(ave): %f' % (mean_shape_ious_ave))
    log_string('eval mean mIoU(ave) (all shapes): %f' % (np.mean(all_shape_ious_ave)))



if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    evaluate()
    LOG_FOUT.close()

