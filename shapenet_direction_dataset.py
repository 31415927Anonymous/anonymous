import os
import os.path
import json
import numpy as np
import sys
import scipy.io as scio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset():
    def __init__(self, root, dir_root, npoints=2500, classification=False, batch_size=16, split='train', state='train', normalize=True, return_cls_label=False, shuffle=None):
        self.npoints = npoints
        self.root = root
        self.dir_root = os.path.join(self.root, dir_root)
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.batch_size = batch_size
        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label
        self.set = state

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in list(self.cat.items())}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            direct = os.path.join(self.dir_root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print(('Unknown split: %s. Exiting..' % (split)))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append([os.path.join(dir_point, token + '.txt'), os.path.join(direct, token + '.mat')])

        self.datapath = []
        for item in self.cat:
            for fn, dfn in self.meta[item]:
                self.datapath.append((item, fn, dfn))

        self.classes = dict(list(zip(self.cat, list(range(len(self.cat))))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

        if shuffle is None:
            if split in ['train', 'trainval']:
                self.shuffle = True
            else:
                self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data, batch_normal, batch_seg, batch_direc):
        batch_data_nor = np.concatenate([batch_data,batch_normal], axis=2)
        rotated_data, rotated_dir = batch_data_nor, batch_direc

        jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        idx = np.arange(batch_data.shape[1])
        return rotated_data[:, idx, 0:3], rotated_data[:, idx, 3:6], batch_seg[:, idx], rotated_dir[:, idx, :]

    def _get_item(self, index):
        if index in self.cache:
            point_set, normal, seg, direc, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            #print(fn[2])
            p = scio.loadmat(fn[2])
            direc = p['vec'].astype(np.float32)
            point_set = data[:, 0:3]
            if self.normalize:
                point_set = pc_normalize(point_set)
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, direc, cls)

        choice = []
        restp = self.npoints
        if self.set == 'eval':
            point_num = point_set.shape[0]
            choice = np.arange(restp) % point_num
        else:
            while (restp>0):
                choice.append(np.random.choice(len(seg), min(len(seg), restp), replace=False))
                restp -= len(seg)
            choice = np.concatenate(choice)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]
        direc = direc[choice, :]

        if self.set == 'eval':
            return point_set, normal, seg, direc, cls, point_num
        if self.classification:
            return point_set, normal, direc, cls
        else:
            if self.return_cls_label:
                return point_set, normal, seg, direc, cls
            else:
                return point_set, normal, seg, direc

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        ''' just for train '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, 3))
        batch_normal = np.zeros((bsize, self.npoints, 3))
        batch_seg = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_direc = np.zeros((bsize, self.npoints, 6))
        batch_cls = np.zeros((bsize,), dtype=np.int32)
        for i in range(bsize):
            if self.return_cls_label:
                ps, normal, seg, direc, cls = self._get_item(self.idxs[i + start_idx])
                batch_cls[i] = cls
            else:
                ps, normal, seg, direc = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = ps
            batch_normal[i] = normal
            batch_seg[i] = seg
            batch_direc[i] = direc
        self.batch_idx += 1
        if augment:
            batch_data, batch_normal, batch_seg, batch_direc = self._augment_batch_data(batch_data, batch_normal, batch_seg, batch_direc)
        if self.return_cls_label:
            return batch_data, batch_normal, batch_seg, batch_direc, batch_cls
        else:
            return batch_data, batch_normal, batch_seg, batch_direc


if __name__ == '__main__':
    d = PartNormalDataset(root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='trainval',
                          npoints=3000, dir_root='directions_seg', return_cls_label=True)
    print((len(d)))

    i = 500
    ps, normal, seg, dire, cls = d[i]
    print((d.datapath[i]))
    print((np.max(seg), np.min(seg)))
    print((ps.shape, seg.shape, normal.shape, dire.shape))
    print(ps)
    print(normal)
    print(cls)

    print(d.shuffle)
    for i in range(10):
        ps, normal, seg, di, cls = d[i]
    print((ps.shape, type(ps), seg))

    print((d.has_next_batch()))
    ps_batch, normal_batch, seg_batch, dir_batch, cls_batch = d.next_batch(True)
    print((ps_batch.shape))
    print((seg_batch.shape))
    print((dir_batch.shape))
    print(cls_batch)

    d = PartNormalDataset(root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', classification=True, dir_root='directions_seg')
    print((len(d)))
    ps, normal, seg, di = d[0]
    print((ps.shape, type(ps), seg.shape, type(seg)))

