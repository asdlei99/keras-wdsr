# https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
import keras.backend as K
from keras.engine.topology import Layer


class PixelShuffler(Layer):
    def __init__(self, size=2):
        super(PixelShuffler, self).__init__()
        self.size = size

    def call(self, inputs):
        input_shape = K.shape(inputs)
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        oh, ow = h * self.size, w * self.size
        oc = c // (self.size * self.size)

        out = K.reshape(inputs, (-1, h, w, self.size, self.size, oc))
        out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
        out = K.reshape(out, (-1, oh, ow, oc))
        return out

    def compute_output_shape(self, input_shape):
        channels = input_shape[3] // self.size // self.size
        return None, None, None, channels
