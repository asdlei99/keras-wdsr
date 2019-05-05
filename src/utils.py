from cv2 import imread, imwrite, resize
import cv2
import numpy as np


def crop(im, size):
    h = np.random.randint(im.shape[0] - size + 1)
    w = np.random.randint(im.shape[1] - size + 1)
    return im[h:h + size, w:w + size]


def rotate(im):
    k = np.random.randint(4)
    return np.rot90(im, k)


def flip(im):
    if np.random.randint(2):
        im = np.flip(im, axis=1)
    return im


def downsampling(im, scale):
    size = im.shape[1] // scale, im.shape[0] // scale
    hidden_scale = np.random.uniform(1, 3)
    hidden_size = int(size[0] / hidden_scale), int(size[1] / hidden_scale)
    radius = np.random.uniform(1, 3)
    im = cv2.GaussianBlur(im, (5, 5), radius)
    im = resize(im, hidden_size)
    return resize(im, size)


def pair(fn, size=96, scale=2):
    im = imread(fn)
    im = crop(im, size)
    im = rotate(im)
    hr = flip(im)
    lr = downsampling(hr, scale)
    return lr, hr


def normalization(im, bgr_mean=127.5):
    return (im - bgr_mean) / 127.5


def denormalization(im, bgr_mean=127.5):
    return im * 127.5 + bgr_mean


if __name__ == '__main__':
    im = imread('a.jpg')
    im = crop(im, 96 * 4)
    imwrite('b.jpg', im)
    im = rotate(im)
    imwrite('c.jpg', im)
    im = flip(im)
    imwrite('d.jpg', im)
    im = downsampling(im, 4)
    imwrite('e.jpg', im)
    lr, hr = pair('a.jpg', size=96 * 6)
    imwrite('lr.jpg', lr)
    imwrite('hr.jpg', hr)
    im = normalization(hr)
    im = denormalization(im)
    imwrite('f.jpg', im)
    print(im.dtype)
    im[:100, :100] = 300
    im[100:200, 100:200] = -100
    imwrite('g.jpg', im)
