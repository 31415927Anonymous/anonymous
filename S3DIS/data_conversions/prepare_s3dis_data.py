#!/usr/bin/python3
'''Prepare Data for S3DIS Segmentation Task.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import numpy as np
from datetime import datetime
import scipy.io as scio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--max_point_num', '-m', help='Max point number of each sample', type=int, default=8192)
    parser.add_argument('--block_size', '-b', help='Block size', type=float, default=0.3)
    parser.add_argument('--grid_size', '-g', help='Grid size', type=float, default=0.03)
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')

    args = parser.parse_args()
    print(args)

    root = args.folder if args.folder else '../../data/S3DIS/prepare_label_rgb'
    dir_root = '../../data/S3DIS/dir_all_d/'
    max_point_num = args.max_point_num

    batch_size = 2048
    data = np.zeros((batch_size, max_point_num, 6))
    direction = np.zeros((batch_size, max_point_num, 6))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
    indices_split_to_full = np.zeros((batch_size, max_point_num), dtype=np.int32)

    data_extra = np.zeros((batch_size, 4096, 6))
    direction_extra = np.zeros((batch_size, 4096, 6))
    data_num_extra = np.zeros((batch_size), dtype=np.int32)
    label_seg_extra = np.zeros((batch_size, 4096), dtype=np.int32)

    mergenum = 0

    for area_idx in range(1, 7):
        folder = os.path.join(root, 'Area_%d' % area_idx)
        datasets = [dataset for dataset in os.listdir(folder)]
        for dataset_idx, dataset in enumerate(datasets):
            dataset_marker = os.path.join(folder, dataset, ".dataset")
            if os.path.exists(dataset_marker):
                print('{}-{}/{} already processed, skipping'.format(datetime.now(), folder, dataset))
                continue
            filename_data = os.path.join(folder, dataset, 'xyzrgb_label.mat')
            print('{}-Loading {}...'.format(datetime.now(), filename_data))
            xyz_label = scio.loadmat(filename_data)
            xyzrgb = xyz_label['xyzrgb']
            labels = xyz_label['label'].astype(int).flatten()

            filename_direction = os.path.join(dir_root, 'Area_%d' % area_idx, dataset, 'vec.mat')
            print('{}-Loading {}...'.format(datetime.now(), filename_direction))
            vec = scio.loadmat(filename_direction)
            vec = vec['vec']

            xyz, rgb = np.split(xyzrgb, [3], axis=-1)
            xyz_min = np.amin(xyz, axis=0, keepdims=True)
            xyz_max = np.amax(xyz, axis=0, keepdims=True)
            xyz_center = (xyz_min + xyz_max) / 2
            xyz_center[0][-1] = xyz_min[0][-1]
            xyz = xyz - xyz_center  # align to room bottom center
            rgb = rgb / 255 - 0.5

            offsets = [('zero', 0.0), ('half', 1.2 / 2)]
            for offset_name, offset in offsets:
                idx_h5 = 0
                idx = 0

                print('{}-Computing block id of {} points...'.format(datetime.now(), xyzrgb.shape[0]))
                xyz_min = np.amin(xyz, axis=0, keepdims=True) - offset
                xyz_max = np.amax(xyz, axis=0, keepdims=True)
                block_size = (args.block_size, args.block_size, 2 * (xyz_max[0, -1] - xyz_min[0, -1]))
                xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int)

                print('{}-Collecting points belong to each block...'.format(datetime.now(), xyzrgb.shape[0]))
                blocks, point_block_indices, block_point_counts = np.unique(xyz_blocks, return_inverse=True,
                                                                            return_counts=True, axis=0)
                block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
                print('{}-{} is split into {} blocks.'.format(datetime.now(), dataset, blocks.shape[0]))

                blockmax = np.amax(blocks, axis=0)
                xblocks = int(blockmax[0] / 5) + 1
                yblocks = int(blockmax[1] / 5) + 1
                blocks_base = [[] for _ in range(xblocks * yblocks)]
                blocks_extra = [[] for _ in range(xblocks * yblocks)]
                for xi in range(blockmax[0]+1):
                    for yj in range(blockmax[1]+1):
                        if sum(abs([xi, yj, 0] - blocks[min((blockmax[1]+1)*xi+yj, blocks.shape[0]-1)])):
                            #print([xi, yj, 0])
                            blocks = np.insert(blocks, (blockmax[1]+1)*xi+yj, [xi, yj, 0], 0)
                            block_point_indices.insert((blockmax[1]+1)*xi+yj, [])
                            #print(len(block_point_indices))
                for xi in range(blockmax[0]+1):
                    for yj in range(blockmax[1]+1):
                        subid = (blockmax[1]+1)*xi+yj
                        bid = yblocks*int(xi/5)+int(yj/5)
                        blocks_base[bid].append(block_point_indices[subid])
                        if ((xi % 5) == 0) & (xi > 0):
                            eid = yblocks * int((xi - 1) / 5) + int(yj / 5)
                            blocks_extra[eid].append(block_point_indices[subid])
                        elif ((xi % 5) == 4) & (xi < blocks[-1][0]):
                            eid = yblocks * int((xi + 1) / 5) + int(yj / 5)
                            blocks_extra[eid].append(block_point_indices[subid])

                        if ((yj % 5) == 0) & (yj > 0):
                            eid = yblocks * int(xi / 5) + int((yj - 1)/5)
                            blocks_extra[eid].append(block_point_indices[subid])
                        elif ((yj % 5) == 4) & (yj < blocks[-1][1]):
                            eid = yblocks * int(xi / 5) + int((yj + 1) / 5)
                            blocks_extra[eid].append(block_point_indices[subid])
                blocks_base = [np.concatenate(ind) for ind in blocks_base]
                blocks_extra = [np.concatenate(ind) for ind in blocks_extra]

                block_to_block_idx_map = dict()
                for block_idx in range(xblocks * yblocks):
                    block = (block_idx // yblocks, block_idx % yblocks)
                    block_to_block_idx_map[(block[0], block[1])] = block_idx

                # merge small blocks into one of their big neighbors
                block_point_count_threshold = max_point_num / 10
                nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
                block_merge_count = 0
                for block_idx in range(xblocks * yblocks):
                    if len(blocks_base[block_idx]) >= block_point_count_threshold:
                        continue

                    block = (blocks[block_idx][0], blocks[block_idx][1])
                    for x, y in nbr_block_offsets:
                        nbr_block = (block[0] + x, block[1] + y)
                        if nbr_block not in block_to_block_idx_map:
                            continue

                        nbr_block_idx = block_to_block_idx_map[nbr_block]
                        if len(blocks_base[nbr_block_idx]) < block_point_count_threshold:
                            continue

                        blocks_base[nbr_block_idx] = np.concatenate([blocks_base[nbr_block_idx],
                                                                     blocks_base[block_idx]], axis=-1)
                        blocks_base[block_idx] = np.array([], dtype=np.int)
                        blocks_extra[nbr_block_idx] = np.concatenate([blocks_extra[nbr_block_idx],
                                                                      blocks_extra[block_idx]], axis=-1)
                        blocks_extra[block_idx] = np.array([], dtype=np.int)

                        blocks_extra[nbr_block_idx] = np.array(list(set(blocks_extra[nbr_block_idx])
                                                                    .difference(set(blocks_base[nbr_block_idx]))))
                        block_merge_count = block_merge_count + 1
                        break
                print('{}-{} of {} blocks are merged.'.format(datetime.now(), block_merge_count, blocks.shape[0]))

                idx_last_non_empty_block = 0
                for block_idx in reversed(range(xblocks * yblocks)):
                    if blocks_base[block_idx].shape[0] != 0:
                        idx_last_non_empty_block = block_idx
                        break

                blocks_base = [ind.astype(np.int) for ind in blocks_base]
                blocks_extra = [ind.astype(np.int) for ind in blocks_extra]

                for block_idx in range(idx_last_non_empty_block + 1):
                    point_indices = blocks_base[block_idx]
                    point_indices_extra = blocks_extra[block_idx]
                    if point_indices.shape[0] == 0:
                        continue

                    block_point_num = point_indices.shape[0]
                    block_split_num = int(math.ceil(block_point_num * 1.0 / max_point_num))
                    point_num_avg = int(math.ceil(block_point_num * 1.0 / block_split_num))
                    point_nums = [point_num_avg] * block_split_num
                    point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
                    starts = [0] + list(np.cumsum(point_nums))

                    np.random.shuffle(point_indices)
                    block_points = xyz[point_indices]
                    block_rgb = rgb[point_indices]
                    block_dir = vec[point_indices]
                    block_labels = labels[point_indices]
                    block_xzyrgb = np.concatenate([block_points, block_rgb], axis=-1)

                    np.random.shuffle(point_indices_extra)
                    block_points_extra = xyz[point_indices_extra]
                    block_rgb_extra = rgb[point_indices_extra]
                    block_dir_extra = vec[point_indices_extra]
                    block_labels_extra = labels[point_indices_extra]
                    block_xzyrgb_extra = np.concatenate([block_points_extra, block_rgb_extra], axis=-1)

                    for block_split_idx in range(block_split_num):
                        start = starts[block_split_idx]
                        point_num = point_nums[block_split_idx]
                        end = start + point_num
                        idx_in_batch = idx % batch_size
                        data[idx_in_batch, 0:point_num, ...] = block_xzyrgb[start:end, :]
                        direction[idx_in_batch, 0:point_num, ...] = block_dir[start:end, :]
                        data_num[idx_in_batch] = point_num
                        label[idx_in_batch] = dataset_idx  # won't be used...
                        label_seg[idx_in_batch, 0:point_num] = block_labels[start:end]
                        indices_split_to_full[idx_in_batch, 0:point_num] = point_indices[start:end]

                        point_num_extra = int(point_num*2.0/4)
                        indice = np.random.choice(len(point_indices_extra), point_num_extra, replace=len(point_indices_extra)<point_num_extra)
                        data_extra[idx_in_batch, 0:point_num_extra, ...] = block_xzyrgb_extra[indice, :]
                        direction_extra[idx_in_batch, 0:point_num_extra, ...] = block_dir_extra[indice, :]
                        data_num_extra[idx_in_batch] = point_num_extra
                        label_seg_extra[idx_in_batch, 0:point_num_extra] = block_labels_extra[indice]

                        if ((idx + 1) % batch_size == 0) or \
                                (block_idx == idx_last_non_empty_block and block_split_idx == block_split_num - 1):
                            item_num = idx_in_batch + 1
                            filename_h5 = os.path.join(folder, dataset, '%s_%d.h5' % (offset_name, idx_h5))
                            print('{}-Saving {}...'.format(datetime.now(), filename_h5))

                            file = h5py.File(filename_h5, 'w')
                            file.create_dataset('data', data=data[0:item_num, ...])
                            file.create_dataset('data_num', data=data_num[0:item_num, ...])
                            file.create_dataset('direction', data=direction[0:item_num, ...])
                            file.create_dataset('label', data=label[0:item_num, ...])
                            file.create_dataset('label_seg', data=label_seg[0:item_num, ...])
                            file.create_dataset('indices_split_to_full', data=indices_split_to_full[0:item_num, ...])
                            file.create_dataset('data_extra', data=data_extra[0:item_num, ...])
                            file.create_dataset('data_num_extra', data=data_num_extra[0:item_num, ...])
                            file.create_dataset('direction_extra', data=direction_extra[0:item_num, ...])
                            file.create_dataset('label_seg_extra', data=label_seg_extra[0:item_num, ...])
                            file.close()

                            if args.save_ply:
                                print('{}-Saving ply of {}...'.format(datetime.now(), filename_h5))
                                filepath_label_ply = os.path.join(folder, dataset, 'ply_label',
                                                                  'label_%s_%d' % (offset_name, idx_h5))
                                data_utils.save_ply_property_batch(data[0:item_num, :, 0:3],
                                                                   label_seg[0:item_num, ...],
                                                                   filepath_label_ply, data_num[0:item_num, ...], 14)

                                filepath_rgb_ply = os.path.join(folder, dataset, 'ply_rgb',
                                                                'rgb_%s_%d' % (offset_name, idx_h5))
                                data_utils.save_ply_color_batch(data[0:item_num, :, 0:3],
                                                                (data[0:item_num, :, 3:] + 0.5) * 255,
                                                                filepath_rgb_ply, data_num[0:item_num, ...])

                            idx_h5 = idx_h5 + 1
                        idx = idx + 1

            # Marker indicating we've processed this dataset
            open(dataset_marker, "w").close()
        print(mergenum)


if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
