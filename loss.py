import tensorflow as tf
import numpy as np


def reconstruction_loss(reconstruct, img, rs):
    """
    :param  reconstruct: B, H, W, 3
    :param  img: B, H, W, 3
    :param  rs: int, the scale of reconstruction loss
    :return loss: int
    """
    diff = reconstruct - img
    loss = diff * diff
    for i in range(len(reconstruct.shape)):
        loss = tf.keras.backend.sum(loss, axis=0)

    return loss
