import json
import os
from copy import deepcopy
from glob import glob
# from random import choice, shuffle
from random import shuffle

# import chainer
import numpy as np
from chainer.dataset import DatasetMixin
# from chainer.dataset.convert import to_device
# from librosa import load
# from librosa.feature import melspectrogram
# from librosa.util import frame
from nnmnkwii.preprocessing import mulaw_quantize
# from pyvad import trim
# from scipy import signal

from .default_settings import DEFAULT


default_dataset_root = os.path.join(DEFAULT['datasetroot'], DEFAULT['npzdir'])


class NPZDataset(DatasetMixin):

    def __init__(self, dataset_root=default_dataset_root, dataset_dir="", param_file="datasetparam.json"):
        if dataset_dir:
            data_dir = os.path.join(dataset_root, dataset_dir)
        else:
            data_dir = dataset_root
        paths = sorted(
            glob(os.path.join(data_dir, '**/*.npz'), recursive=True))
        self._paths = paths
        with open(os.path.join(dataset_root, param_file), 'r') as f:
            load = json.load(f)

        self.mspec_max = load.pop('mspec_max')
        self.pspec_max = load.pop('pspec_max')
        self.mspec_min = load.pop('mspec_min')
        self.pspec_min = load.pop('pspec_min')
        self.upsample =  load.pop('upsample')
        self.params = load 
        
    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        return dict(np.load(path))

    def get_example_from_names(self, names, random=True):
        names = deepcopy(names)
        if random:
            shuffle(names)

        path = None
        for name in names:
            for p in self._paths:
                if name == os.path.basename(p):
                    print(p)
                    path = p
                    break

            if path is not None:
                break

        if path is None:
            raise FileNotFoundError

        return dict(np.load(path))


def transform_for_wnv(indata, spec_max, spec_min=1e-10, length=7680, upsample=80, log_spec=True, softmax=False):
    if length:
        if len(indata['wave']) <= length:
            indata['wave'] = np.pad(
                indata['wave'], (0, length - len(indata['wave']) + 1), 'constant', constant_values=0)
            spec_len = length // upsample
            indata['mspec'] = np.pad(
                indata['mspec'], (0, spec_len - len(indata['mspec'])), 'constant', constant_values=spec_min)
            index = 0
        else:
            index = np.random.randint(0, len(indata['wave']) - length)

        index = (index // upsample) * upsample
        indata['wave'] = indata['wave'][index:index + length + 1]
        indata['mspec'] = indata['mspec'][index //
                                          upsample:(index + length) // upsample]

    indata['mspec'] = np.clip(indata['mspec'], spec_min, spec_max)
    if log_spec:
        indata['mspec'] = np.log10(indata['mspec'])
        shift = np.log10(spec_min)
        scale = np.log10(spec_max) - np.log10(spec_min)
        indata['mspec'] = (indata['mspec'] - shift) / scale
    else:
        shift = spec_min
        scale = spec_max - spec_min
        indata['mspec'] = (indata['mspec'] - shift) / scale

    rdict = {}
    rdict['lc'] = indata['mspec'].astype('float32').T
    rdict['wave'] = (indata('wave').reshape(
        1, -1) / (2.0**15 - 1)).astype('float32')
    if softmax:
        rdict['wave'] = mulaw_quantize(rdict['wave']).astype('int32')
    return rdict


def transform_for_ssrn(indata, mspec_max, pspec_max, length=None, mspec_min=1e-10, pspec_min=1e-10, log_spec=True):
    if length:
        if len(indata['mspec']) <= length:
            indata['mspec'] = np.pad(
                indata['mspec'], (0, length - len(indata['mspec'])), 'constant', constant_values=mspec_min)
            indata['pspec'] = np.pad(
                indata['pspec'], (0, length - len(indata['pspec'])), 'constant', constant_values=pspec_min)
            index = 0
        else:
            index = np.random.randint(0, len(indata['mspec']) - length)

        indata['mspec'] = indata['mspec'][index:index + length]
        indata['pspec'] = indata['pspec'][index:index + length]

    indata['pspec'] = np.clip(indata['pspec'], pspec_min, pspec_max)
    indata['mspec'] = np.clip(indata['mspec'], mspec_min, mspec_max)

    if log_spec:
        indata['mspec'] = np.log10(indata['mspec'])
        shift = np.log10(mspec_min)
        scale = np.log10(mspec_max) - np.log10(mspec_min)
        indata['mspec'] = (indata['mspec'] - shift) / scale

        indata['pspec'] = np.log10(indata['pspec'])
        shift = np.log10(pspec_min)
        scale = np.log10(pspec_max) - np.log10(pspec_min)
        indata['pspec'] = (indata['pspec'] - shift) / scale
    else:
        shift = mspec_min
        scale = mspec_max - mspec_min
        indata['mspec'] = (indata['mspec'] - shift) / scale

        shift = pspec_min
        scale = pspec_max - pspec_min
        indata['pspec'] = (indata['pspec'] - shift) / scale

    return indata['mspec'].astype('float32'), indata['pspec'].astype('float32')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = NPZDataset(length=7680)
    print(len(dataset))
    print(np.random.randint(len(dataset)))
    x = dataset.get_example(np.random.randint(len(dataset)))
    print(x)
