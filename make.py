# -*- coding: utf-8 -*-
import argparse
import json
import os
from glob import glob
from multiprocessing import Value
from random import choice

import chainer
import numpy as np
from chainer.dataset import DatasetMixin
from chainer.dataset.convert import to_device
from joblib import Parallel, delayed
from librosa import load
from librosa.feature import melspectrogram
from librosa.util import frame
from nnmnkwii.preprocessing import mulaw_quantize
from pyvad import trim
from scipy import signal

from default_settings import DEFAULT

mspec_max = Value('f', -10000000000.0)
pspec_max = Value('f', -10000000000.0)
mspec_min = Value('f', 100000000000.0)
pspec_min = Value('f', 100000000000.0)
s_upsample = Value('i', 0)


def process(wavpath, wav_dir, npz_dir, override, kwargs):
    print(wavpath)
    path, name = os.path.split(os.path.relpath(wavpath, wav_dir))
    path = os.path.join(npz_dir, path)
    name, _ = os.path.splitext(name)
    savepath = os.path.join(path, name) + '.npz'

    if override:
        if os.path.exists(savepath):
            print('already exists.')
            data = np.load(savepath)
            mspec_max.value = max((data['mspec'].max(), mspec_max.value))
            mspec_min.value = min((data['mspec'].min(), mspec_min.value))
            pspec_max.value = max((data['pspec'].max(), pspec_max.value))
            pspec_min.value = min((data['pspec'].min(), pspec_min.value))
            return

    data, fs = load(wavpath, sr=kwargs['fs'], res_type=kwargs['res_type'])

    if np.max(np.abs(data)) > 1:
        data = data / np.max(np.abs(data)) * 0.9
    trimed = trim(data, fs, vad_mode=3)

    if trimed is not None:
        if len(trimed) < kwargs['minimum_len']:
            return
        _, _, Zxx = signal.stft(
            trimed, fs=kwargs['fs'], window=kwargs['window'], nperseg=kwargs['nperseg'], noverlap=kwargs['noverlap'])
        pspec = np.abs(Zxx)
        pspec[pspec < kwargs['spec_threshold']] = kwargs['spec_threshold']
        mspec = melspectrogram(
            sr=kwargs['fs'], S=pspec, n_mels=kwargs['nmels'], power=1.0)
        mspec[mspec < kwargs['spec_threshold']] = kwargs['spec_threshold']
        pspec = np.log10(pspec).T.astype('float32')
        mspec = np.log10(mspec).T.astype('float32')
        upsample = kwargs['fs'] // (1000 // kwargs['frame_period'])
        s_upsample.value = upsample
        length = (len(trimed) // upsample) * upsample
        trimed = trimed[:length]
        mspec = mspec[:length // upsample]
        pspec = pspec[:length // upsample]
        mspec_max.value = max((mspec.max(), mspec_max.value))
        mspec_min.value = min((mspec.min(), mspec_min.value))
        pspec_max.value = max((pspec.max(), pspec_max.value))
        pspec_min.value = min((pspec.min(), pspec_min.value))
        trimed = (trimed * (2**15 - 1)).astype('int16')
        os.makedirs(path, exist_ok=True)
        np.savez(savepath,
                    wave=trimed.astype('int16'), mspec=mspec.astype('float32'), pspec=pspec.astype('float32'))


def make_npz(override, **kwargs):
    npz_dir = os.path.join(kwargs['datasetroot'], kwargs['npzdir'])
    wav_dir = os.path.join(kwargs['datasetroot'], kwargs['wavdir'])
    wavpaths = sorted(
        glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True))

    try:
        
        Parallel(n_jobs=-1)([delayed(process)(wavpath, wav_dir, npz_dir, override, kwargs)
                             for wavpath in wavpaths])
        """
        for wavpath in wavpaths:
            process(wavpath, wav_dir, npz_dir, override, kwargs)
        """
        

    except KeyboardInterrupt:
        pass

    save = dict(mspec_max=float(mspec_max.value),
                mspec_min=float(mspec_min.value),
                pspec_max=float(pspec_max.value),
                pspec_min=float(pspec_min.value),
                upsample=int(s_upsample.value))
    save.update(kwargs)
    with open(os.path.join(npz_dir, 'datasetparam.json'), 'w') as f:
        json.dump(save, f, indent=1, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Wavenet.')
    parser.add_argument('--override', '-or', action='store_false',
                        help='Use GPU')
    args = vars(parser.parse_args())
    make_npz(args['override'] ,**DEFAULT)
