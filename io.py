import numpy as np
from scipy.io.wavfile import write


def write_wav(filename, rate, data):
    data = np.squeeze(data)
    if data.dtype != 'int16':
        if np.abs(data).max() > 1.0:
            data = data / np.abs(data).max() * 0.99

        data = (data * (2**(16 - 1) - 1)).astype('int16')

    write(filename, rate, data)
