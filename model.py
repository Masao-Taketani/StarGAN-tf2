import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU,\
     ReLU, Conv2DTranspose, Dropout, concatenate, ZeroPadding2D, Input
from tensorflow.keras.activations import tanh


class ResidualBlock(Layer):
    # Define Re block
    def __init__(self,
                 filters, 
                 size=3, 
                 strides=1, 
                 padding="same", 
                 name="residual_block"):

        super(ResidualBlock, self).__init__()
        self.size = size
        self.conv2d_1 = Conv2D(filters, 
                               size, 
                               strides, 
                               padding=padding,
                               use_bias=False)
        self.instance_norm_1 = InstanceNormalization()
        self.ReLU = ReLU()
        self.conv2d_2 = Conv2D(filters,
                               size,
                               strides,
                               padding=padding,
                               use_bias=False)
        self.instance_norm_2 = InstanceNormalization()

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.instance_norm_1(x)
        x = self.ReLU(x)
        x = self.conv2d_2(x)
        x = self.instance_norm_2(x)
        return x + inputs


class InstanceNormalization(Layer):
    """
    Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
    Applies Instance Normalization for each channel in each data sample in a
    batch.
    Args:
        epsilon: a small positive decimal number to avoid dividing by 0
    """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(name="scale",
                                     shape=input_shape[-1:],
                                     initializer="glorot_uniform",
                                     trainable=True)

        self.offset = self.add_weight(name="offset",
                                      shape=input_shape[-1:],
                                      initializer="zeros",
                                      trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset