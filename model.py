import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU,\
     ReLU, Conv2DTranspose, Dropout, concatenate, ZeroPadding2D, Input
from tensorflow.keras.activations import tanh


INPUT_SHAPE = (128, 128, 3)

def get_norm_layer(norm_type):
    if norm_type.lower() == "batchnorm":
        return BatchNormalization()
    elif norm_type.lower() == "instancenorm":
        return InstanceNormalization()
    else:
        raise ValueError("arg `norm_type` has to be either batchnorm "
                         "or instancenorm. What you specified is "
                         "{}".format(norm_type))


def get_activation(activation):
    if activation.lower() == "relu":
        return ReLU()
    elif activation.lower() == "lrelu":
        return LeakyReLU(0.01)
    elif activation.lower() == "tanh":
        return tanh
    else:
        raise ValueError("arg `norm_type` has to be either relu "
                         "or tanh. What you specified is "
                         "{}".format(norm_type))


class ResidualBlock(Layer):
    # Define Re block
    def __init__(self,
                 filters, 
                 size=3, 
                 strides=1, 
                 padding="same", 
                 name="residual_block"):

        super(ResidualBlock, self).__init__(name=name)
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


class Downsample(Layer):
    """
     Conv2D -> BatchNorm(or InstanceNorm) -> LeakyReLU
     Args:
        filters: number of filters
           size: filter size
      norm_type: normalization type. Either "batchnorm", "instancenorm" or None
           name: name of the layer
    Return:
        Downsample functional model
    """

    def __init__(self, 
                 filters, 
                 size,
                 strides=1,
                 padding="same",
                 norm_type="instancenorm",
                 activation="relu",
                 name="downsample"):

        super(Downsample, self).__init__(name=name)
        self.norm_type = norm_type
        use_bias = False
        if self.norm_type:
            self.norm_layer = get_norm_layer(norm_type)
        else:
            use_bias = True
        self.conv2d = Conv2D(filters,
                             size,
                             strides=strides,
                             padding=padding,
                             use_bias=use_bias)
        self.activation = get_activation(activation)

    def call(self, inputs):
        x = self.conv2d(inputs)
        if self.norm_type:
            x = self.norm_layer(x)
        x = self.activation(x)

        return x


class Upsample(Layer):
    """
    Conv2DTranspose -> BatchNorm(or InstanceNorm) -> Dropout -> ReLU
     Args:
        filters: number of filters
           size: filter size
      norm_type: normalization type. Either "batchnorm", "instancenorm" or None
  apply_dropout: If True, apply the dropout layer
           name: name of the layer
    Return:
        Upsample functional model
    """
    def __init__(self, 
                 filters, 
                 size,
                 strides,
                 padding,
                 norm_type="instancenorm",
                 apply_dropout=False,
                 activation="relu",
                 name="upsample"):

        super(Upsample, self).__init__(name=name)
        self.norm_type = norm_type
        use_bias = False
        if self.norm_type:
            self.norm_layer = get_norm_layer(norm_type)
        else:
            use_bias = True
        self.apply_dropout = apply_dropout
        self.conv2dtranspose = Conv2DTranspose(filters,
                                               size,
                                               strides=strides,
                                               padding=padding,
                                               use_bias=use_bias)
        if apply_dropout:
            self.dropout = Dropout(0.5)
        self.activation = get_activation(activation)

    def call(self, inputs):
        x = self.conv2dtranspose(inputs)
        if self.norm_type:
            x = self.norm_layer(x)
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        return x


class Discriminator(Model):
    """
    Referred from StarGAN paper(https://arxiv.org/abs/1711.09020).
    [Network Architecture]
    Input Layer:
        (h, w, 3) → (h/2, w/2, 64) CONV-(N64, K4x4, S2, P1), Leaky ReLU
    Hidden Layer:
        (h/2, w/2, 64) → (h/4, w/4, 128)        CONV-(N128, K4x4, S2, P1), Leaky ReLU
        (h/4, w/4, 128) → (h/8, w/8, 256)       CONV-(N256, K4x4, S2, P1), Leaky ReLU
        (h/8, w/8, 256) → (h/16, w16 , 512)     CONV-(N512, K4x4, S2, P1), Leaky ReLU
        (h/16, w/16, 512) → (h/32, w/32, 1024)  CONV-(N1024, K4x4, S2, P1), Leaky ReLU
        (h/32, w/32, 1024) → (h/64, w/64, 2048) CONV-(N2048, K4x4, S2, P1), Leaky ReLU
    Output Layer:
        (Dsrc) (h/64, w/64 , 2048) → (h/64, w/64, 1) CONV-(N1, K3x3, S1, P1)
        (Dcls) (h/64, w/64 , 2048) → (1, 1, nd)      CONV-(N(nd), K h/64 x w/64 , S1, P0)

    Args:
        norm_type: normalization type. Either "batchnorm", "instancenorm" or None
        
    Return:
        Discriminator model
    """
    def __init__(self,
                 c_dim,
                 first_filters=64,
                 size=4,
                 norm_type=None,
                 img_size=128,
                 name="discriminator"):

        super(Discriminator, self).__init__(name=name)
        self.downsample_1 = Downsample(filters=first_filters, 
                                       size=size,
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_1")
        self.downsample_2 = Downsample(filters=first_filters * 2, 
                                       size=size,
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_2")
        self.downsample_3 = Downsample(filters=first_filters * 4, 
                                       size=size,
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_3")
        self.downsample_4 = Downsample(filters=first_filters * 8, 
                                       size=size, 
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_4")
        self.downsample_5 = Downsample(filters=first_filters * 16, 
                                       size=size, 
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_5")
        self.downsample_6 = Downsample(filters=first_filters * 32, 
                                       size=size, 
                                       strides=2,
                                       padding="same",
                                       norm_type=norm_type,
                                       activation="lrelu",
                                       name="d_downsample_6")
        self.conv2d_src = Conv2D(filters=1, 
                                 kernel_size=3, 
                                 strides=1,
                                 padding="same")
        self.conv2d_cls = Conv2D(filters=c_dim,
                                 kernel_size=img_size//64,
                                 strides=1,
                                 padding="valid")

    def call(self, inputs):
        x = self.downsample_1(inputs)
        x = self.downsample_2(x) 
        x = self.downsample_3(x)
        x = self.downsample_4(x)
        x = self.downsample_5(x)
        x = self.downsample_6(x)
        out_src = self.conv2d_src(x)
        out_cls = self.conv2d_cls(x)

        return out_src, out_cls

    def summary(self):
        x = Input(shape=INPUT_SHAPE)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class Generator(Model):
    """
    Referred from StarGAN paper(https://arxiv.org/abs/1711.09020).
    [Network Architecture]
    Down-sampling:
        (h, w, 3 + nc) → (h, w, 64)       CONV-(N64, K7x7, S1, P3), IN, ReLU
        (h, w, 64) → (h/2, w/2, 128)      CONV-(N128, K4x4, S2, P1), IN, ReLU
        (h/2, w/2, 128) → (h/4, w/4, 256) CONV-(N256, K4x4, S2, P1), IN, ReLU
    Bottleneck:
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
        (h/4, w/4, 256) → (h/4, w/4, 256) Residual Block: CONV-(N256, K3x3, S1, P1), IN, ReLU
    Up-sampling:
        (h/4, w/4, 256) → (h/2, w/2, 128) DECONV-(N128, K4x4, S2, P1), IN, ReLU
        (h/2, w/2, 128) → (h, w, 64) DECONV-(N64, K4x4, S2, P1), IN, ReLU
        (h, w, 64) → (h, w, 3) CONV-(N3, K7x7, S1, P3), Tanh

    Args:
    output_channels: number of output channels
          norm_type: normalization type. Either "batchnorm", "instancenorm" or None
    Return:
        Generator model
    """
    def __init__(self,
                 first_filters=64,
                 output_channels=3,
                 norm_type="instancenorm", 
                 name="generator"):

        super(Generator, self).__init__(name=name)
        self.downsample_1 = Downsample(filters=first_filters, 
                                       size=7,
                                       strides=1,
                                       padding="valid",
                                       norm_type=norm_type, 
                                       name="g_downsample_1")
        self.downsample_2 = Downsample(filters=first_filters*2, 
                                       size=4,
                                       strides=2,
                                       norm_type=norm_type, 
                                       name="g_downsample_2")
        self.downsample_3 = Downsample(filters=first_filters*4, 
                                       size=4,
                                       strides=2,
                                       norm_type=norm_type, 
                                       name="g_downsample_3")
        self.residualblock_1 = ResidualBlock(first_filters*4, name="g_residualblock_1")
        self.residualblock_2 = ResidualBlock(first_filters*4, name="g_residualblock_2")
        self.residualblock_3 = ResidualBlock(first_filters*4, name="g_residualblock_3")
        self.residualblock_4 = ResidualBlock(first_filters*4, name="g_residualblock_4")
        self.residualblock_5 = ResidualBlock(first_filters*4, name="g_residualblock_5")
        self.residualblock_6 = ResidualBlock(first_filters*4, name="g_residualblock_6")
        self.upsample_1 = Upsample(filters=first_filters*2, 
                                   size=4,
                                   strides=2,
                                   padding="same",
                                   norm_type=norm_type,
                                   name="g_upsample_1")
        self.upsample_2 = Upsample(filters=first_filters, 
                                   size=4,
                                   strides=2,
                                   padding="same",
                                   norm_type=norm_type,
                                   name="g_upsample_2")
        self.last_conv2d = Conv2D(filters=output_channels, 
                                  kernel_size=7,
                                  strides=1,
                                  padding="valid",
                                  activation="tanh",
                                  name="g_last_conv2d")

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        x = self.residualblock_1(x)
        x = self.residualblock_2(x)
        x = self.residualblock_3(x)
        x = self.residualblock_4(x)
        x = self.residualblock_5(x)
        x = self.residualblock_6(x)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        result = self.last_conv2d(x)

        return result

    def summary(self):
        x = Input(shape=INPUT_SHAPE)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


if __name__ == "__main__":
    test_disc = Discriminator(c_dim=5)
    print("Check Discriminator's model architecture")
    test_disc.summary()

    test_gen = Generator()
    print("Check Generator's model architecture")
    test_gen.summary()