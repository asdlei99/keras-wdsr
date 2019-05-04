import numpy as np
from keras import backend as K


def psnr(hr, sr, max_val=2):
    mse = K.mean(K.square(hr - sr))
    return 10.0 / np.log(10) * K.log(max_val ** 2 / mse)
