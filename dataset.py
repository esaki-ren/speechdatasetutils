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




class NPZDataset(DatasetMixin):

    def __init__(self, mode, length=7680):
        paths = sorted(
            glob(os.path.join(NPZROOT, 'data*.npz'), recursive=True))
        self._paths = paths
        load = np.load(os.path.join(NPZROOT, 'minmax.npz'))
        self.shift = load['min']
        self.scale = load['max'] - load['min']
        self.upsample = load['upsample']
        self.length = length
        self.mode = mode

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        load = dict(np.load(path))
        index = np.random.randint(0, len(load['wave']) - self.length)
        index = (index // self.upsample) * self.upsample
        wave = load.pop('wave')[index:index + self.length + 1]
        load['wave'] = (wave.reshape(
            1, -1) / (2.0**15 - 1)).astype('float32')
        lc = load.pop('lc')[..., index //
                            self.upsample:(index + self.length) // self.upsample]
        load['lc'] = (lc - self.shift) / self.scale
        if self.mode == 'softmax':
            load['wave'] = mulaw_quantize(load['wave']).astype('int32')
        return load



if __name__ == '__main__':
    make_npz()
