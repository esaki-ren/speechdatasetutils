import json
import os
from glob import glob
from random import choice

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

from .default_settings import DEFAULT


default_dataset_root = os.path.join(DEFAULT['datasetroot'], DEFAULT['npzdir'])


class NPZDataset(DatasetMixin):

    def __init__(self, dataset_root=default_dataset_root, length=7680, spec_mode='conv', mode='mixture', keydict=dict(wave='wave', lc='mspec')):
        paths = sorted(
            glob(os.path.join(dataset_root, '**/*.npz'), recursive=True))
        self._paths = paths
        with open(os.path.join(default_dataset_root, 'datasetparam.json'), 'r') as f:
            load = json.load(f)
        self.m_shift = load['mspec_min']
        self.m_scale = load['mspec_max'] - load['mspec_min']
        self.p_shift = load['pspec_min']
        self.p_scale = load['pspec_max'] - load['pspec_min']
        self.upsample = load['upsample']
        self.length = length
        self.mode = mode
        self.spec_mode = spec_mode
        self.keydict = keydict

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        load = dict(np.load(path))
        if self.length:
            if len(load['wave']) < self.length:
                load['wave'] = np.pad(
                    load['wave'], (0, self.length - len(load['wave']) + 1), 'constant', constant_values=0)
                spec_len = self.length // self.upsample
                load['mspec'] = np.pad(
                    load['mspec'], (0, spec_len - len(load['mspec'])), 'constant', constant_values=self.m_min)
                load['pspec'] = np.pad(
                    load['pspec'], (0, spec_len - len(load['pspec'])), 'constant', constant_values=self.p_min)
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
            print(i, rkey, key)
            if key is 'mspec':
                rdict[rkey] = ((load.pop('mspec') - self.m_shift) / self.m_scale).astype('float32')
                if self.spec_mode is 'conv':
                    rdict[rkey] = rdict[rkey].T
            elif key is 'pspec':
                rdict[rkey] = ((load.pop('pspec') - self.p_shift) / self.p_scale).astype('float32')
                if self.spec_mode is 'conv':
                    rdict[rkey] = rdict[rkey].T
            elif key is 'wave':
                print(i, 'wav process')
                rdict[rkey] = (load.pop('wave').reshape(
                    1, -1) / (2.0**15 - 1)).astype('float32')
                if self.mode == 'softmax':
                    rdict[rkey] = mulaw_quantize(rdict[rkey]).astype('int32')
        
        print(i, rdict.keys())
        return rdict


if __name__ == '__main__':
    NPZDataset(length=7680).get_example(1)
