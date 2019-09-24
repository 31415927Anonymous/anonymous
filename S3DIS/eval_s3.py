import argparse
from datetime import datetime
import h5py
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='seg_d_conv_rgb', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log_eval]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--direction', default='directions_seg', help='directions')
parser.add_argument('--area', '-a', type=int, default=5, help='val area number')
parser.add_argument('--repeat_num', '-r', type=int, default=16, help='val area number')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

REPEAT_NUM = FLAGS.repeat_num
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
NUM_CLASSES = 13

DATA_PATH = '../data/S3DIS/prepare_label_rgb'

classes = {'clutter': 0,
           'ceiling': 1,
           'floor': 2,
           'wall': 3,
           'beam': 4,
           'column': 5,
           'door': 6,
           'window': 7,
           'table': 8,
           'chair': 9,
           'sofa': 10,
           'bookcase': 11,
           'board': 12}

labeltoclass = {}
for key in classes.keys():
    labeltoclass[classes[key]] = key


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, direc_pl = MODEL.placeholder_inputs(REPEAT_NUM, None)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            print("--- Get model and loss")
            pred = MODEL.get_model(pointclouds_pl, direc_pl, is_training_pl)
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
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}

        eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False

    loss_sum = 0
    iou_max = []
    iou_ave = []
    total_correct_max = 0
    total_correct_ave = 0
    total_seen = 0

    part_ious_max = [[] for _ in range(NUM_CLASSES)]
    part_ious_ave = [[] for _ in range(NUM_CLASSES)]

    C_max = np.zeros([NUM_CLASSES, 3])
    C_ave = np.zeros([NUM_CLASSES, 3])

    AREA = os.path.join(DATA_PATH, 'Area_%d' % FLAGS.area)
    for ROOM in os.listdir(AREA):
        print(ROOM)
        ROOM_PATH_ZERO = os.path.join(AREA, ROOM, 'zero_0.h5')
        ROOM_PATH_HALF = os.path.join(AREA, ROOM, 'half_0.h5')
        ROOM_PATH_WHOLE = os.path.join(AREA, ROOM, 'xyzrgb_label.mat')

        points = []
        direction = []
        point_nums = []
        labels_seg = []
        indices_split_to_full = []

        points_extra = []
        point_nums_extra = []
        labels_seg_extra = []
        direction_extra = []

        data = h5py.File(ROOM_PATH_ZERO)
        points.append(data['data'][...].astype(np.float32))
        direction.append(data['direction'][...].astype(np.float32))
        point_nums.append(data['data_num'][...].astype(np.int32))
        labels_seg.append(data['label_seg'][...].astype(np.int64))
        indices_split_to_full.append(data['indices_split_to_full'][...].astype(np.int64))
        points_extra.append(data['data_extra'][...].astype(np.float32))
        direction_extra.append(data['direction_extra'][...].astype(np.float32))
        point_nums_extra.append(data['data_num_extra'][...].astype(np.int32))
        labels_seg_extra.append(data['label_seg_extra'][...].astype(np.int64))

        data = h5py.File(ROOM_PATH_HALF)
        points.append(data['data'][...].astype(np.float32))
        direction.append(data['direction'][...].astype(np.float32))
        point_nums.append(data['data_num'][...].astype(np.int32))
        labels_seg.append(data['label_seg'][...].astype(np.int64))
        indices_split_to_full.append(data['indices_split_to_full'][...].astype(np.int64))
        points_extra.append(data['data_extra'][...].astype(np.float32))
        direction_extra.append(data['direction_extra'][...].astype(np.float32))
        point_nums_extra.append(data['data_num_extra'][...].astype(np.int32))
        labels_seg_extra.append(data['label_seg_extra'][...].astype(np.int64))

        del data

        points = np.concatenate(points, axis=0)
        direction = np.concatenate(direction, axis=0)
        point_nums = np.concatenate(point_nums, axis=0)
        labels_seg = np.concatenate(labels_seg, axis=0)
        indices_split_to_full = np.concatenate(indices_split_to_full, axis=0)
        points_extra = np.concatenate(points_extra, axis=0)
        point_nums_extra = np.concatenate(point_nums_extra, axis=0)
        labels_seg_extra = np.concatenate(labels_seg_extra, axis=0)
        direction_extra = np.concatenate(direction_extra, axis=0)

        loss_room_sum = 0
        room_max_logits = []
        room_ave_logits = []
        for blockid in range(points.shape[0]):
            block_point = points[blockid]  # Nx6
            block_dir = direction[blockid]
            block_num = point_nums[blockid]
            block_label = labels_seg[blockid]
            block_point_extra = points_extra[blockid]  # Nx6
            block_dir_extra = direction_extra[blockid]
            block_num_extra = point_nums_extra[blockid]
            block_label_extra = labels_seg_extra[blockid]

            choice = np.arange(REPEAT_NUM * NUM_POINT) % block_num
            choice = np.transpose(np.reshape(choice, [NUM_POINT, REPEAT_NUM]))
            int(NUM_POINT/2)
            choice_extra = np.arange(REPEAT_NUM * int(NUM_POINT/2)) % block_num_extra
            choice_extra = np.transpose(np.reshape(choice_extra, [int(NUM_POINT/2), REPEAT_NUM]))
            # ---------------------------------------------------------------------
            feed_dict = {ops['pointclouds_pl']: np.concatenate([block_point[choice], block_point_extra[choice_extra]], axis=1),
                         ops['dir_pl']: np.concatenate([block_dir[choice], block_dir_extra[choice_extra]], axis=1),
                         ops['labels_pl']: np.concatenate([block_label[choice], block_label_extra[choice_extra]], axis=1),
                         ops['is_training_pl']: is_training, }
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            # ---------------------------------------------------------------------
            loss_room_sum += loss_val

            pred_val = pred_val[:, :NUM_POINT, :]

            pred_val = pred_val.transpose([1, 0, 2])

            cur_pred_logits = np.reshape(pred_val, (REPEAT_NUM * NUM_POINT, NUM_CLASSES))
            pred_logits1 = map(lambda i: cur_pred_logits[i::block_num], range(block_num))

            store = [np.reshape(np.amax(logits, axis=0), [1, 13]) for logits in pred_logits1]
            block_max_logits = np.concatenate(store, axis=0)
            pred_logits2 = map(lambda i: cur_pred_logits[i::block_num], range(block_num))
            store = [np.reshape(np.average(logits, axis=0), [1, 13]) for logits in pred_logits2]
            block_ave_logits = np.concatenate(store, axis=0)
            room_max_logits.append(block_max_logits)
            room_ave_logits.append(block_ave_logits)
        loss_room_sum /= points.shape[0]
        print('loss: {}, point: {}'.format(loss_room_sum, points.shape[0]))
        room_max_logits = np.concatenate(room_max_logits, axis=0)  # allblockpoint_num * 13
        room_ave_logits = np.concatenate(room_ave_logits, axis=0)
        room_ind = [ind for ind in map(lambda i: indices_split_to_full[i, 0:point_nums[i]], range(points.shape[0]))]
        room_ind = np.concatenate(room_ind, axis=0)  # ind to full of logits

        del points
        del direction
        del labels_seg
        gc.collect()

        # load gt label
        gt_label = loadmat(ROOM_PATH_WHOLE)['label']
        gt_label = np.reshape(gt_label, [-1])

        ind = room_ind < gt_label.shape[0]
        room_ind = room_ind[ind]
        room_max_logits = room_max_logits[ind]
        room_ave_logits = room_ave_logits[ind]
        te, indices, counts = np.unique(room_ind, return_inverse=True, return_counts=True, axis=0)
        point_indices = np.split(np.argsort(indices), np.cumsum(counts[:-1]))  # list of gt_label.shape[0] nparray
        room_max_val = np.zeros((gt_label.shape[0]), dtype=int)
        room_ave_val = np.zeros((gt_label.shape[0]), dtype=int)
        room_max_val[te] = [pred for pred in
                            map(lambda i: np.argmax(np.amax(room_max_logits[point_indices[i]], axis=0)),
                                range(te.shape[0]))]
        room_ave_val[te] = [pred for pred in
                            map(lambda i: np.argmax(np.average(room_ave_logits[point_indices[i]], axis=0)),
                                range(te.shape[0]))]
        correct_max = np.sum(room_max_val == gt_label)
        correct_ave = np.sum(room_ave_val == gt_label)
        total_correct_max += correct_max
        total_correct_ave += correct_ave
        total_seen += gt_label.shape[0]
        print('max: {}, ave: {}, seen: {}'.format(correct_max, correct_ave, gt_label.shape[0]))

        del room_max_logits
        del room_ave_logits
        gc.collect()

        room_ious_max = []
        room_ious_ave = []
        print(str(datetime.now()))
        for l in range(NUM_CLASSES):
            if (np.sum(room_max_val == l) == 0) and (np.sum(gt_label == l) == 0):
                # part_ious_max[l] = 1.0
                print('no {}'.format(labeltoclass[l]))
            else:
                a = room_max_val == l
                b = gt_label == l
                m = map(lambda i: a[i] & b[i], range(gt_label.shape[0]))
                inter = np.sum([i for i in m])
                part_ious_max[l].append(inter / float(np.sum(a) + np.sum(b) - inter))
                room_ious_max.append(inter / float(np.sum(a) + np.sum(b) - inter))
                C_max[l, 0] += np.sum(b)
                C_max[l, 1] += np.sum(a)
                C_max[l, 2] += np.sum(inter)

            if (np.sum(room_ave_val == l) == 0) and (np.sum(gt_label == l) == 0):
                # part_ious_ave[l] = 1.0
                continue
            else:
                a = room_ave_val == l
                b = gt_label == l
                m = map(lambda i: a[i] & b[i], range(gt_label.shape[0]))
                inter = np.sum([i for i in m])
                part_ious_ave[l].append(inter / float(np.sum(a) + np.sum(b) - inter))
                room_ious_ave.append(inter / float(np.sum(a) + np.sum(b) - inter))
                C_ave[l, 0] += np.sum(b)
                C_ave[l, 1] += np.sum(a)
                C_ave[l, 2] += np.sum(inter)
        print(str(datetime.now()))
        iou_max.append(np.mean(room_ious_max))
        iou_ave.append(np.mean(room_ious_ave))
        loss_sum += loss_room_sum
    loss_sum /= len(os.listdir(AREA))
    iou_max = np.mean(iou_max, axis=0)
    iou_ave = np.mean(iou_ave, axis=0)


    log_string('eval mean loss: %f' % loss_sum)
    iou = 0
    for cat in classes.keys():
        l = classes[cat]
        iou_l = float(C_max[l, 2]/(C_max[l, 0]+C_max[l, 1]-C_max[l, 2]))
        iou += iou_l
        log_string('eval mIoU(max) of %s:\t %f' % (cat, np.mean(part_ious_max[classes[cat]])))
        log_string('eval IoU(max) of %s:\t %f' % (cat, iou_l))
    log_string('eval mean mIoU(max): %f' % (iou_max))
    log_string('eval mean IoU(max): %f' % (iou/NUM_CLASSES))
    log_string('eval accuracy(max): %f' % (total_correct_max / float(total_seen)))
    log_string('--------------------------------------------')
    iou = 0
    for cat in classes.keys():
        l = classes[cat]
        iou_l = float(C_ave[l, 2] / (C_ave[l, 0] + C_ave[l, 1] - C_ave[l, 2]))
        iou += iou_l
        log_string('eval mIoU(average) of %s:\t %f' % (cat, np.mean(part_ious_ave[classes[cat]])))
        log_string('eval IoU(average) of %s:\t %f' % (cat, iou_l))
    log_string('eval mean mIoU(average): %f' % (iou_ave))
    log_string('eval mean IoU(average): %f' % (iou / NUM_CLASSES))
    log_string('eval accuracy(average): %f' % (total_correct_ave / float(total_seen)))


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
