import numpy as np
import tensorflow as tf
from tensorflow import keras

from capsule_layers import *


class MultiReconstruction(keras.Model):
    """
    Input is capsule: B, H, W, N, A (N is number of classes)
    Label is tensor: B, H, W, N (N is number of classes)

    How to create mask tensor ?
    1. manipulate label tensor following
        Label reshape:-> B, H ,W, 1, N, Label append in number of A and concatinate in axis 3:-> B, H, W, A, N
        Label reshape:-> B, H, W, N*A
    2. multiply input capsule and label
        Mask tensor = input capsule * label

    Mask tensor is used to use capsule at (y, x, n) if label at (y, x, n) is 1.

    Outpus Shape: B, H, W, 3 (RGB)

    """
    def __init__(self, img_channels):
        super(MultiReconstruction, self).__init__(name='multi_reconstruction')
        self.reconv1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
                                     name='reconv1', kernel_initializer=tf.keras.initializers.he_normal())
        self.reconv2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', activation='relu',
                                     name='reconv2', kernel_initializer=tf.keras.initializers.he_normal())
        self.reconv3 = layers.Conv2D(filters=img_channels, kernel_size=1, padding='same', activation='sigmoid',
                                     name='reconv3', kernel_initializer=tf.keras.initializers.glorot_normal())

    def call(self, input_capsule, lbl_tensor):
        _, h, w, n, a = input_capsule.shape
        input_capsule_reshaped = tf.reshape(input_capsule, (-1, h, w, n*a))

        # Label append in number of A and is concatenated in axis 3
        lbl_tensor_list = []
        lbl_tensor = tf.expand_dims(lbl_tensor, axis=-1)
        for i in range(a):
            lbl_tensor_list.append(lbl_tensor)

        lbl_proliferated = tf.concat(lbl_tensor_list, axis=-1)
        lbl_proliferated = tf.reshape(lbl_proliferated, (-1, h, w, n*a))

        masked_seg_capsule = input_capsule_reshaped * lbl_proliferated
        h = self.reconv1(masked_seg_capsule)
        h = self.reconv2(h)
        h = self.reconv3(h)

        return h



class MultiSegCaps(keras.Model):

    def __init__(self, n_class, img_channels=3):
        super(MultiSegCaps, self).__init__(name='segcaps')
        self.n_class = n_class

        # Set up layers
        self.conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')

        # Layer 1: Primary Capsule: Conv cap with routing 1
        self.primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                        routings=1, name='primarycaps')

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                        routings=3, name='conv_cap_2_1')

        # Layer 2: Convolutional Capsule
        self.conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                        routings=3, name='conv_cap_2_2')

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_3_1')

        # Layer 3: Convolutional Capsule
        self.conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                        routings=3, name='conv_cap_3_2')

        # Layer 4: Convolutional Capsule
        self.conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                        routings=3, name='conv_cap_4_1')

        # Layer 1 Up: Deconvolutional Capsule
        self.deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_1_1')

        # Skip connection
        self.up_1 = layers.Concatenate(axis=-2, name='up_1')

        # Layer 1 Up: Deconvolutional Capsule
        self.deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                          padding='same', routings=3, name='deconv_cap_1_2')

        # Layer 2 Up: Deconvolutional Capsule
        self.deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_2_1')

        # Skip connection
        self.up_2 = layers.Concatenate(axis=-2, name='up_2')

        # Layer 2 Up: Deconvolutional Capsule
        self.deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                          padding='same', routings=3, name='deconv_cap_2_2')

        # Layer 3 Up: Deconvolutional Capsule
        self.deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_3_1')

        # Skip connection
        self.up_3 = layers.Concatenate(axis=-2, name='up_3')

        # Layer 4: Convolutional Capsule: 1x1
        self.seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=n_class, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='seg_caps')

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        self.out_seg = MultiLength(seg=True, name='out_seg')

        # This Object is
        self.reconstruct = MultiReconstruction(img_channels)

    def call(self, inputs, lbl):
        conv1 = self.conv1(inputs)
        # Reshape tensor to be 1 capsule x atoms
        _, h, w, c = conv1.shape
        conv1_reshaped = tf.reshape(conv1, (-1, h, w, 1, c))
        primary_caps = self.primary_caps(conv1_reshaped)

        conv_cap_2_1 = self.conv_cap_2_1(primary_caps)
        conv_cap_2_2 = self.conv_cap_2_2(conv_cap_2_1)

        conv_cap_3_1 = self.conv_cap_3_1(conv_cap_2_2)
        conv_cap_3_2 = self.conv_cap_3_2(conv_cap_3_1)

        conv_cap_4_1 = self.conv_cap_4_1(conv_cap_3_2)

        deconv_cap_1_1 = self.deconv_cap_1_1(conv_cap_4_1)

        up_1 = tf.concat((deconv_cap_1_1, conv_cap_3_1), axis=-2)  # Skip connection
        deconv_cap_1_2 = self.deconv_cap_1_2(up_1)
        deconv_cap_2_1 = self.deconv_cap_2_1(deconv_cap_1_2)

        up_2 = tf.concat((deconv_cap_2_1, conv_cap_2_1), axis=-2)
        deconv_cap_2_2 = self.deconv_cap_2_2(up_2)
        deconv_cap_3_1 = self.deconv_cap_3_1(deconv_cap_2_2)

        up_3 = tf.concat((deconv_cap_3_1, conv1_reshaped), axis=-2)
        seg_caps = self.seg_caps(up_3)

        # output
        out_seg = self.out_seg(seg_caps)
        reconstuct = self.reconstruct(seg_caps, lbl)

        return out_seg, reconstuct


#    def compute_output_shape(self, input_shape):
#        print(input_shape)
#        exit(1)
#        shape = tf.TensorShape(input_shape).as_list()
#        shape[-1] = self.n_class
#        return shape


if __name__ == "__main__":
    tf.enable_eager_execution()
    tf.executing_eagerly()
    model = MultiSegCaps(4)
    x = tf.ones((2, 128, 128, 3))
    out = model(x)
    print("success")
