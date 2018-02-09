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

WAVROOT = os.path.expanduser(os.path.join('~', 'dataset', 'wav', ))
DATASETROOT = os.path.expanduser(os.path.join('~', 'dataset', 'wav', 'npy'))
NPZROOT = os.path.expanduser(os.path.join('~', 'dataset', 'wav', 'npz'))


def make_npz(Fs=16000):
    os.makedirs(NPZROOT, exist_ok=True)
    WINDOW = 'hann'
    FRAME_PERIOD = 5  # [ms]
    NSHIFT = 16000 * FRAME_PERIOD // 1000
    NPERSEG = 960
    NMELS = 80
    NOVERLAP = NPERSEG - NSHIFT
    assert signal.check_COLA(WINDOW, NPERSEG, NOVERLAP)

    wavpaths = sorted(
        glob(os.path.join(WAVROOT, '**', '*.wav'), recursive=True))
    counter = 0
    lc_max = -10000000000.0
    lc_min = 100000000000.0
    try:
        for path in wavpaths:
            print(path)
            data, fs = load(path, sr=Fs, res_type='scipy')

            if np.max(np.abs(data)) > 1:
                data = data / np.max(np.abs(data)) * 0.9
            trimed = trim(data, fs, vad_mode=3)

            if trimed is not None:
                f, t, Zxx = signal.stft(
                    trimed, fs=Fs, window=WINDOW, nperseg=NPERSEG, noverlap=NOVERLAP)
                pspec = np.abs(Zxx)**2
                mspec = melspectrogram(sr=Fs, S=pspec, n_mels=NMELS)
                mspec = np.log10(mspec).T.astype('float32')
                upsample = Fs // (1000 // FRAME_PERIOD)
                length = (len(trimed)// upsample)  * upsample
                trimed = trimed[:length]
                mspec = mspec[:length // upsample]
                lc_max = max((mspec.max(), lc_max))
                lc_min = min((mspec.min(), lc_min))
                trimed = (trimed * (2**15 - 1)).astype('int16')
                np.savez(os.path.join(NPZROOT, 'data_' + str(counter)),
                         wave=trimed.astype('int16'), lc=mspec.T.astype('float32'))
                counter += 1
    except KeyboardInterrupt:
        pass
    np.savez(os.path.join(NPZROOT, 'minmax'), max=np.array(
        lc_max).astype('float32'), min=np.array(lc_min).astype('float32'), upsample=upsample)


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
