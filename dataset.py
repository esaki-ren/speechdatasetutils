import json
import os
from copy import deepcopy
from glob import glob
from random import choice, shuffle

import chainer
import numpy as np
from chainer.dataset import DatasetMixin
from chainer.dataset.convert import to_device
from librosa import load
from librosa.feature import melspectrogram
from librosa.util import frame
from nnmnkwii.preprocessing import mulaw_quantize
from pyvad import trim
from scipy import signal

from default_settings import DEFAULT


default_dataset_root = os.path.join(DEFAULT['datasetroot'], DEFAULT['npzdir'])


class NPZDataset(DatasetMixin):

    def __init__(self, dataset_root=default_dataset_root, dataset_dir="", param_file="datasetparam.json", length=7680, spec_mode='conv', mode='mixture', log_spec=True, spec_thr=None, keydict=dict(wave='wave', lc='mspec')):
        if dataset_dir:
            data_dir = os.path.join(dataset_root, dataset_dir)
        else:
            data_dir = dataset_root
        paths = sorted(
            glob(os.path.join(data_dir, '**/*.npz'), recursive=True))
        self._paths = paths
        with open(os.path.join(dataset_root, param_file), 'r') as f:
            load = json.load(f)
        
        mspec_max = load['mspec_max']
        pspec_max = load['pspec_max']
        if 
        mspec_min = load['mspec_min']
        pspec_min = load['pspec_min']
        self.m_shift = load['mspec_min']
        self.m_scale = load['mspec_max'] - load['mspec_min']
        self.p_shift = load['pspec_min']
        self.p_scale = load['pspec_max'] - load['pspec_min']
        self.upsample = load['upsample']
        self.length = length
        self.mode = mode
        self.spec_mode = spec_mode
        self.keydict = keydict
        self.log_spec = log_spec
        self.spec_thr = spec_thr
        

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        return self.npzprocess(path)

    def npzprocess(self, path):
        load = dict(np.load(path))
        if self.length:
            if len(load['wave']) <= self.length:
                load['wave'] = np.pad(
                    load['wave'], (0, self.length - len(load['wave']) + 1), 'constant', constant_values=0)
                spec_len = self.length // self.upsample
                load['mspec'] = np.pad(
                    load['mspec'], (0, spec_len - len(load['mspec'])), 'constant', constant_values=self.m_shift)
                load['pspec'] = np.pad(
                    load['pspec'], (0, spec_len - len(load['pspec'])), 'constant', constant_values=self.p_shift)
                index = 0
            else:
                index = np.random.randint(0, len(load['wave']) - self.length)

            index = (index // self.upsample) * self.upsample
            load['wave'] = load['wave'][index:index + self.length + 1]
            load['mspec'] = load['mspec'][index //
                                          self.upsample:(index + self.length) // self.upsample]
            load['pspec'] = load['pspec'][index //
                                          self.upsample:(index + self.length) // self.upsample]

        rdict = {}
        for rkey, key in self.keydict.items():
            if key == 'mspec':
                rdict[rkey] = ((load.pop('mspec') - self.m_shift) /
                               self.m_scale).astype('float32')
                if self.spec_mode == 'conv':
                    rdict[rkey] = rdict[rkey].T
            elif key == 'pspec':
                rdict[rkey] = ((load.pop('pspec') - self.p_shift) /
                               self.p_scale).astype('float32')
                if self.spec_mode == 'conv':
                    rdict[rkey] = rdict[rkey].T
            elif key == 'wave':
                rdict[rkey] = (load.pop('wave').reshape(
                    1, -1) / (2.0**15 - 1)).astype('float32')
                if self.mode == 'softmax':
                    rdict[rkey] = mulaw_quantize(rdict[rkey]).astype('int32')

    def get_example_from_names(self, names, random=True):
        names = deepcopy(names)
        if random:
            shuffle(names)

        path = None
        for name in names:
            for p in self._paths:
                if name in p:
                    path = p
                    break
            break

        if path is None:
            return None

        return self.npzprocess(path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = NPZDataset(length=7680)
    print(len(dataset))
    print(np.random.randint(len(dataset)))
    x = dataset.get_example(np.random.randint(len(dataset)))
    print(x)
