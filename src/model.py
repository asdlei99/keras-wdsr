from keras.layers import Add, Conv2D, Input, Activation
from keras.models import Model

from pixel_shuffler import PixelShuffler


def res_block_b(x_in, filters):
    expand = 6
    linear = 0.8
    x = Conv2D(filters * expand, 1)(x_in)
    x = Activation('relu')(x)
    x = Conv2D(int(filters * linear), 1)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x_in, x])
    return x


def wdsr(scale, filters, n_resblocks, res_block):
    x_in = Input(shape=(None, None, 3))

    m = Conv2D(filters, 3, padding='same')(x_in)
    for i in range(n_resblocks):
        m = res_block(m, filters)
    m = Conv2D(3 * scale ** 2, 3, padding='same')(m)
    m = PixelShuffler(scale)(m)

    s = Conv2D(3 * scale ** 2, 5, padding='same')(x_in)
    s = PixelShuffler(scale)(s)
    x = Add()([m, s])
    return Model(x_in, x)


def wdsr_b(scale=2, filters=32, n_resblocks=8):
    return wdsr(scale, filters, n_resblocks, res_block_b)


if __name__ == '__main__':
    import utils
    model = wdsr_b(scale=4)
    lr, hr = utils.pair('a.jpg', scale=4)
    lr.shape = (1,) + lr.shape
    print(lr.shape, hr.shape)
    im = model.predict(lr)
    print(im.shape)
    model.save_weights('model.h5')
