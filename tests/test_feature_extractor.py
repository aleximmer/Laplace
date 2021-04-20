import torch
import torch.nn as nn
import torchvision.models as models

from laplace.feature_extractor import FeatureExtractor


class CNN(nn.Module):
    def __init__(self, num_classes):
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            # Input shape (3, 64, 64)
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # Output shape (6, 60, 60)
            nn.ReLU(),
            # Output shape (6, 30, 30)
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 16 * 16,
                      out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300,
                      out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84,
                      out_features=num_classes)
        )

        self.conv2 = nn.Sequential(
            # Input shape (6, 30, 30)
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # Output shape (16, 26, 26)
            nn.ReLU(),
            # Output shape (16, 13, 13)
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        # print("x = self.conv1(x)   ", x.shape, "  ", torch.Size)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18()
    elif model_name == 'alexnet':
        model = models.alexnet()
    elif model_name == 'vgg16':
        model = models.vgg16()
    elif model_name == 'squeezenet':
        model = models.squeezenet1_0()
    elif model_name == 'densenet':
        model = models.densenet161()
    elif model_name == 'inception':
        model = models.inception_v3(init_weights=True)
    elif model_name == 'googlenet':
        model = models.googlenet(init_weights=True)
    elif model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0()
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2()
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large()
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small()
    elif model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d()
    elif model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2()
    elif model_name == 'mnasnet':
        model = models.mnasnet1_0()
    elif model_name == 'switchedCNN':
        model = CNN(10)
    elif model_name == 'sequential':
        model = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6*64*64, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
    else:
        raise ValueError(f'{model_name} is not supported.')
    return model


@torch.no_grad()
def test_feature_extractor():
    # all torchvision classifcation models but 'squeezenet' (no linear last layer)
    # + model where modules are initilaized in wrong order + nn.Sequential model
    model_names = ['resnet18', 'alexnet', 'vgg16', 'densenet', 'inception',
                   'googlenet', 'shufflenet', 'mobilenet_v2', 'mobilenet_v3_large',
                   'mobilenet_v3_small', 'resnext50_32x4d', 'wide_resnet50_2',
                   'mnasnet', 'switchedCNN', 'sequential']

    # to test the last_layer_name argument
    # last_layer_names = ['fc', 'classifier.6', 'classifier.6', 'classifier', 'fc',
    #                     'fc', 'fc', 'classifier.1', 'classifier.3', 'classifier.3',
    #                     'fc', 'fc', 'classifier.1', 'fc.4', '5']

    for model_name in model_names:
        # generate random test input
        if model_name == 'inception':
            x = torch.randn(1, 3, 299, 299)
        else:
            x = torch.randn(1, 3, 64, 64)

        model = get_model(model_name)
        model.eval()

        # extract features and get output
        feature_extractor = FeatureExtractor(model)
        out, features = feature_extractor.forward_with_features(x)

        # test if it worked
        last_layer = feature_extractor.last_layer
        out2 = last_layer(features)
        assert (out == out2).all().item()
