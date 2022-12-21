from torch import nn
from deepobs.pytorch.testproblems.testproblems_utils import tfconv2d, \
    tfmaxpool2d, _truncated_normal_init


def get_model(model_name, ds_train):
    if model_name == 'CNN':
        return CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True)
    elif model_name == 'AllCNN':
        return CIFAR100Net(ds_train.channels, ds_train.K)
    else:
        raise ValueError('Invalid model name')


class CIFAR100Net(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    called `net_cifar100_allcnnc`
    """

    def __init__(self, in_channels=3, n_out=100):
        super(CIFAR100Net, self).__init__()
        assert n_out in (10, 100)
        self.output_size = n_out
        # only supported for these

        # self.add_module('dropout1', nn.Dropout(p=0.2))

        self.add_module('conv1', tfconv2d(
            in_channels=in_channels, out_channels=96, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', tfconv2d(
            in_channels=96, out_channels=96, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3', tfconv2d(in_channels=96, out_channels=96, kernel_size=3,
                                          stride=(2, 2), tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())

        # self.add_module('dropout2', nn.Dropout(p=0.5))

        self.add_module('conv4', tfconv2d(
            in_channels=96, out_channels=192, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu4', nn.ReLU())
        self.add_module('conv5', tfconv2d(
            in_channels=192, out_channels=192, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu5', nn.ReLU())
        self.add_module('conv6', tfconv2d(in_channels=192, out_channels=192, kernel_size=3,
                                          stride=(2, 2), tf_padding_type='same'))
        self.add_module('relu6', nn.ReLU())

        # self.add_module('dropout3', nn.Dropout(p=0.5))

        self.add_module('conv7', tfconv2d(
            in_channels=192, out_channels=192, kernel_size=3))
        self.add_module('relu7', nn.ReLU())
        self.add_module('conv8', tfconv2d(
            in_channels=192, out_channels=192, kernel_size=1, tf_padding_type='same'))
        self.add_module('relu8', nn.ReLU())
        self.add_module('conv9', tfconv2d(
            in_channels=192, out_channels=n_out, kernel_size=1, tf_padding_type='same'))

        self.add_module('avg', nn.AvgPool2d(kernel_size=(6, 6)))
        self.add_module('flatten', nn.Flatten())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)


class CIFAR10Net(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    In Deepobs called `net_cifar10_3c3d`
    """

    def __init__(self, in_channels=3, n_out=10, use_tanh=False):
        super(CIFAR10Net, self).__init__()
        self.output_size = n_out
        activ = nn.Tanh if use_tanh else nn.ReLU

        self.add_module('conv1', tfconv2d(
            in_channels=in_channels, out_channels=64, kernel_size=5))
        self.add_module('relu1', nn.ReLU())
        self.add_module('maxpool1', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv2', tfconv2d(
            in_channels=64, out_channels=96, kernel_size=3))
        self.add_module('relu2', nn.ReLU())
        self.add_module('maxpool2', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv3', tfconv2d(
            in_channels=96, out_channels=128, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())
        self.add_module('maxpool3', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('flatten', nn.Flatten())

        self.add_module('dense1', nn.Linear(
            in_features=3 * 3 * 128, out_features=512))
        self.add_module('relu4', activ())
        self.add_module('dense2', nn.Linear(in_features=512, out_features=256))
        self.add_module('relu5', activ())
        self.add_module('dense3', nn.Linear(in_features=256, out_features=n_out))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)