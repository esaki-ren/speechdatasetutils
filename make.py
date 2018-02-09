import os
from glob import glob
from random import choice
import json

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


def make_npz(**kwargs):
    npz_dir = os.path.join(kwargs['datasetroot'], kwargs['npzdir'])
    wav_dir = os.path.join(kwargs['datasetroot'], kwargs['wavdir'])
    wavpaths = sorted(
        glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True))
    mspec_max = pspec_max = -10000000000.0
    mspec_min = pspec_min = 100000000000.0
    try:
        for wavpath in wavpaths:
            print(wavpath)
            path, name = os.path.split(os.path.relpath(wavpath, wav_dir))
            path = os.path.join(npz_dir, path)
            name, _ = os.path.splitext(name)
            
            data, fs = load(wavpath, sr=kwargs['fs'], res_type=kwargs['res_type'])
            
            if np.max(np.abs(data)) > 1:
                data = data / np.max(np.abs(data)) * 0.9
            trimed = trim(data, fs, vad_mode=3)

            if trimed is not None:
                _, _, Zxx = signal.stft(
                    trimed, fs=kwargs['fs'], window=kwargs['window'], nperseg=kwargs['nperseg'], noverlap=kwargs['noverlap'])
                pspec = np.abs(Zxx)
                pspec[pspec < kwargs['spec_threshold']] = kwargs['spec_threshold']
                mspec = melspectrogram(sr=kwargs['fs'], S=pspec, n_mels=kwargs['nmels'], power=1.0)
                mspec[mspec < kwargs['spec_threshold']] = kwargs['spec_threshold']
                pspec = np.log10(pspec).T.astype('float32')
                mspec = np.log10(mspec).T.astype('float32')
                upsample = kwargs['fs'] // (1000 // kwargs['frame_period'])
                length = (len(trimed)// upsample)  * upsample
                trimed = trimed[:length]
                mspec = mspec[:length // upsample]
                pspec = pspec[:length // upsample]
                mspec_max = max((mspec.max(), mspec_max))
                mspec_min = min((mspec.min(), mspec_min))
                pspec_max = max((pspec.max(), pspec_max))
                pspec_min = min((pspec.min(), pspec_min))
                trimed = (trimed * (2**15 - 1)).astype('int16')
                os.makedirs(path, exist_ok=True)
                np.savez(os.path.join(path, name),
                         wave=trimed.astype('int16'), mspec=mspec.astype('float32'), pspec=pspec.astype('float32') )
            
    except KeyboardInterrupt:
        pass
    
    save = dict(mspec_max=float(mspec_max), 
                mspec_min=float(mspec_min), 
                pspec_max=float(pspec_max), 
                pspec_min=float(pspec_min), 
                upsample=upsample)
    save.update(kwargs)
    with open(os.path.join(npz_dir, 'datasetparam.json'), 'w') as f:
        json.dump(save, f, indent=1, sort_keys=True)
            

if __name__ == '__main__':
    make_npz(**DEFAULT)
