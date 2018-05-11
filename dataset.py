import numpy as np


def transform_for_ssrn(indata, mspec_max, pspec_max, length=None, mspec_min=1e-10, pspec_min=1e-10, log_spec=True):
    if 'espec' in indata:
        spec = 'espec'
    else:
        spec = spec
    if length:
        if len(indata['mspec']) <= length:
            indata['mspec'] = np.pad(
                indata['mspec'], (0, length - len(indata['mspec'])), 'constant', constant_values=mspec_min)
            indata[spec] = np.pad(
                indata[spec], (0, length - len(indata[spec])), 'constant', constant_values=pspec_min)
            index = 0
        else:
            index = np.random.randint(0, len(indata['mspec']) - length)

        indata['mspec'] = indata['mspec'][index:index + length]
        indata[spec] = indata[spec][index:index + length]

    indata[spec] = np.clip(indata[spec], pspec_min, pspec_max)
    indata['mspec'] = np.clip(indata['mspec'], mspec_min, mspec_max)

    if log_spec:
        indata['mspec'] = np.log10(indata['mspec'])
        shift = np.log10(mspec_min)
        scale = np.log10(mspec_max) - np.log10(mspec_min)
        indata['mspec'] = (indata['mspec'] - shift) / scale

        indata[spec] = np.log10(indata[spec])
        shift = np.log10(pspec_min)
        scale = np.log10(pspec_max) - np.log10(pspec_min)
        indata[spec] = (indata[spec] - shift) / scale
    else:
        shift = mspec_min
        scale = mspec_max - mspec_min
        indata['mspec'] = (indata['mspec'] - shift) / scale

        shift = pspec_min
        scale = pspec_max - pspec_min
        indata[spec] = (indata[spec] - shift) / scale

    return indata['mspec'].astype('float32'), indata[spec].astype('float32') * 0.9 + 0.05
