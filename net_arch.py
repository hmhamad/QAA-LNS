import src.python.layers as layers
from src.python.utils import relu
from collections import OrderedDict


def load_architecture(network_name, nClasses, inputshape, arithmetic):
    """
    layers_dict (OrderedDict): Dictionary containing the ordered layers of the network layers_dict[0]...layers_dict[nlayers-1] .
    Should follow the format:
        1) layers_dict[0] is the first layer in the network, with input of shape "inputshape"
        2) Last layer is always a FullyConnectedLayerWithSoftmax() with number of units equal to "nClasses"
        3) Each layers' input shape is the previous layer output shape
    """

    layers_dict = OrderedDict()
    if network_name == "testcnn":
        layers_dict[0] = layers.Convolutional2DLayer(
            nFilters=6,
            kernelsize=5,
            stride=1,
            padding="valid",
            act=relu,
            inputshape=inputshape,
            arithmetic=arithmetic,
        )
        layers_dict[1] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            inputshape=layers_dict[0].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[2] = layers.Convolutional2DLayer(
            nFilters=16,
            kernelsize=5,
            stride=1,
            padding="valid",
            act=relu,
            inputshape=layers_dict[1].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[3] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            inputshape=layers_dict[2].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[4] = layers.FlattenLayer(
            inputshape=layers_dict[3].outputshape, arithmetic=arithmetic
        )
        layers_dict[5] = layers.FullyConnectedLayer(
            n_out=120, n_in=layers_dict[4].outputshape, act=relu, arithmetic=arithmetic
        )
        # layers_dict[6] = layers.DropoutLayer(p=0.5,inputshape=layers_dict[5].outputshape,arithmetic=arithmetic)
        layers_dict[6] = layers.FullyConnectedLayer(
            n_out=84, n_in=layers_dict[5].outputshape, act=relu, arithmetic=arithmetic
        )
        # layers_dict[8] = layers.DropoutLayer(p=0.5,inputshape=layers_dict[7].outputshape,arithmetic=arithmetic)
        layers_dict[7] = layers.FullyConnectedLayerWithSoftmax(
            n_out=nClasses, n_in=layers_dict[6].outputshape, arithmetic=arithmetic
        )

    elif network_name == "ResNet18":
        # Stem
        layers_dict[0] = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=7,
            stride=2,
            padding=3,
            act=None,
            inputshape=inputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layers_dict[1] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[0].outputshape, act="relu", arithmetic=arithmetic
        )
        layers_dict[2] = layers.MaxPooling2DLayer(
            inputshape=layers_dict[1].outputshape,
            kernelsize=3,
            stride=2,
            padding=1,
            arithmetic=arithmetic,
        )

        # Conv2_x
        layer1_0_conv1 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[2].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer1_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer1_0_conv2 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer1_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer1_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer1_0_conv1),
                (1, layer1_0_bn1),
                (2, layer1_0_conv2),
                (3, layer1_0_bn2),
            ]
        )
        layers_dict[3] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        layer1_1_conv1 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[3].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer1_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer1_1_conv2 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer1_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer1_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer1_1_conv1),
                (1, layer1_1_bn1),
                (2, layer1_1_conv2),
                (3, layer1_1_bn2),
            ]
        )
        layers_dict[4] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Conv3_x
        layer2_0_conv1 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=2,
            padding=1,
            act=None,
            inputshape=layers_dict[4].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer2_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer2_0_conv2 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer2_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer2_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer2_0_conv1),
                (1, layer2_0_bn1),
                (2, layer2_0_conv2),
                (3, layer2_0_bn2),
            ]
        )
        layers_dict[5] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=2, arithmetic=arithmetic
        )

        layer2_1_conv1 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[5].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer2_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer2_1_conv2 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer2_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer2_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer2_1_conv1),
                (1, layer2_1_bn1),
                (2, layer2_1_conv2),
                (3, layer2_1_bn2),
            ]
        )
        layers_dict[6] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Conv4_x
        layer3_0_conv1 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=2,
            padding=1,
            act=None,
            inputshape=layers_dict[6].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer3_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer3_0_conv2 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer3_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer3_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer3_0_conv1),
                (1, layer3_0_bn1),
                (2, layer3_0_conv2),
                (3, layer3_0_bn2),
            ]
        )
        layers_dict[7] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=2, arithmetic=arithmetic
        )

        layer3_1_conv1 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[7].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer3_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer3_1_conv2 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer3_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer3_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer3_1_conv1),
                (1, layer3_1_bn1),
                (2, layer3_1_conv2),
                (3, layer3_1_bn2),
            ]
        )
        layers_dict[8] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Conv5_x
        layer4_0_conv1 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=2,
            padding=1,
            act=None,
            inputshape=layers_dict[8].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer4_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer4_0_conv2 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer4_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer4_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer4_0_conv1),
                (1, layer4_0_bn1),
                (2, layer4_0_conv2),
                (3, layer4_0_bn2),
            ]
        )
        layers_dict[9] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=2, arithmetic=arithmetic
        )

        layer4_1_conv1 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[9].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer4_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer4_1_conv2 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer4_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer4_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer4_1_conv1),
                (1, layer4_1_bn1),
                (2, layer4_1_conv2),
                (3, layer4_1_bn2),
            ]
        )
        layers_dict[10] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Classifier
        layers_dict[11] = layers.GlobalAveragePooling2DLayer(
            inputshape=layers_dict[10].outputshape, arithmetic=arithmetic
        )
        layers_dict[12] = layers.FullyConnectedLayerWithSoftmax(
            n_out=nClasses, n_in=layers_dict[11].outputshape, arithmetic=arithmetic
        )

    elif network_name == "CifarResNet18":
        """ReNet18 variant for Cifar10 and Cifar100 datasets"""
        # Stem
        layers_dict[0] = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=inputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layers_dict[1] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[0].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv2_x
        layer1_0_conv1 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[1].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer1_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer1_0_conv2 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer1_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer1_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer1_0_conv1),
                (1, layer1_0_bn1),
                (2, layer1_0_conv2),
                (3, layer1_0_bn2),
            ]
        )
        layers_dict[2] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        layer1_1_conv1 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[2].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer1_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer1_1_conv2 = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer1_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer1_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer1_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer1_1_conv1),
                (1, layer1_1_bn1),
                (2, layer1_1_conv2),
                (3, layer1_1_bn2),
            ]
        )
        layers_dict[3] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Conv3_x
        layer2_0_conv1 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=2,
            padding=1,
            act=None,
            inputshape=layers_dict[3].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer2_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer2_0_conv2 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer2_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer2_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer2_0_conv1),
                (1, layer2_0_bn1),
                (2, layer2_0_conv2),
                (3, layer2_0_bn2),
            ]
        )
        layers_dict[4] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=2, arithmetic=arithmetic
        )

        layer2_1_conv1 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[4].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer2_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer2_1_conv2 = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer2_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer2_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer2_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer2_1_conv1),
                (1, layer2_1_bn1),
                (2, layer2_1_conv2),
                (3, layer2_1_bn2),
            ]
        )
        layers_dict[5] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Conv4_x
        layer3_0_conv1 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=2,
            padding=1,
            act=None,
            inputshape=layers_dict[5].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer3_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer3_0_conv2 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer3_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer3_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer3_0_conv1),
                (1, layer3_0_bn1),
                (2, layer3_0_conv2),
                (3, layer3_0_bn2),
            ]
        )
        layers_dict[6] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=2, arithmetic=arithmetic
        )

        layer3_1_conv1 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[6].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer3_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer3_1_conv2 = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer3_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer3_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer3_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer3_1_conv1),
                (1, layer3_1_bn1),
                (2, layer3_1_conv2),
                (3, layer3_1_bn2),
            ]
        )
        layers_dict[7] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Conv5_x
        layer4_0_conv1 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=2,
            padding=1,
            act=None,
            inputshape=layers_dict[7].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_0_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer4_0_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer4_0_conv2 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer4_0_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_0_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer4_0_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer4_0_conv1),
                (1, layer4_0_bn1),
                (2, layer4_0_conv2),
                (3, layer4_0_bn2),
            ]
        )
        layers_dict[8] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=2, arithmetic=arithmetic
        )

        layer4_1_conv1 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layers_dict[8].outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_1_bn1 = layers.BatchNormalizationLayer(
            inputshape=layer4_1_conv1.outputshape, act="relu", arithmetic=arithmetic
        )
        layer4_1_conv2 = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            inputshape=layer4_1_bn1.outputshape,
            bias=False,
            arithmetic=arithmetic,
        )
        layer4_1_bn2 = layers.BatchNormalizationLayer(
            inputshape=layer4_1_conv2.outputshape, act=None, arithmetic=arithmetic
        )
        residual_layers = OrderedDict(
            [
                (0, layer4_1_conv1),
                (1, layer4_1_bn1),
                (2, layer4_1_conv2),
                (3, layer4_1_bn2),
            ]
        )
        layers_dict[9] = layers.ResidualBlock(
            residual_layers, expansion=1, stride=1, arithmetic=arithmetic
        )

        # Classifier
        layers_dict[10] = layers.GlobalAveragePooling2DLayer(
            inputshape=layers_dict[9].outputshape, arithmetic=arithmetic
        )
        layers_dict[11] = layers.FullyConnectedLayerWithSoftmax(
            n_out=nClasses, n_in=layers_dict[10].outputshape, arithmetic=arithmetic
        )

    elif network_name == "VGG16":
        # Conv1
        layers_dict[0] = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=inputshape,
            arithmetic=arithmetic,
        )
        layers_dict[1] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[0].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv2
        layers_dict[2] = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[1].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[3] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[2].outputshape, act="relu", arithmetic=arithmetic
        )

        # MaxPool1
        layers_dict[4] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[3].outputshape,
            arithmetic=arithmetic,
        )

        # Conv3
        layers_dict[5] = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[4].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[6] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[5].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv4
        layers_dict[7] = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[6].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[8] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[7].outputshape, act="relu", arithmetic=arithmetic
        )

        # MaxPool2
        layers_dict[9] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[8].outputshape,
            arithmetic=arithmetic,
        )

        # Conv5
        layers_dict[10] = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[9].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[11] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[10].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv6
        layers_dict[12] = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[11].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[13] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[12].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv7
        layers_dict[14] = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[13].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[15] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[14].outputshape, act="relu", arithmetic=arithmetic
        )

        # MaxPool3
        layers_dict[16] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[15].outputshape,
            arithmetic=arithmetic,
        )

        # Conv8
        layers_dict[17] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[16].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[18] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[17].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv9
        layers_dict[19] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[18].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[20] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[19].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv10
        layers_dict[21] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[20].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[22] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[21].outputshape, act="relu", arithmetic=arithmetic
        )

        # MaxPool4
        layers_dict[23] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[22].outputshape,
            arithmetic=arithmetic,
        )

        # Conv11
        layers_dict[24] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[23].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[25] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[24].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv12
        layers_dict[26] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[25].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[27] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[26].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv13
        layers_dict[28] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[27].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[29] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[28].outputshape, act="relu", arithmetic=arithmetic
        )

        # MaxPool5
        layers_dict[30] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[29].outputshape,
            arithmetic=arithmetic,
        )

        # Classifier
        layers_dict[31] = layers.GlobalAveragePooling2DLayer(
            inputshape=layers_dict[30].outputshape, arithmetic=arithmetic
        )

        layers_dict[32] = layers.FullyConnectedLayer(
            n_out=4096,
            n_in=layers_dict[31].outputshape,
            act="relu",
            arithmetic=arithmetic,
        )
        layers_dict[33] = layers.DropoutLayer(
            p=0.5, inputshape=layers_dict[32].outputshape, arithmetic=arithmetic
        )

        layers_dict[34] = layers.FullyConnectedLayer(
            n_out=4096,
            n_in=layers_dict[33].outputshape,
            act="relu",
            arithmetic=arithmetic,
        )
        layers_dict[35] = layers.DropoutLayer(
            p=0.5, inputshape=layers_dict[34].outputshape, arithmetic=arithmetic
        )

        layers_dict[36] = layers.FullyConnectedLayerWithSoftmax(
            n_out=nClasses, n_in=layers_dict[35].outputshape, arithmetic=arithmetic
        )

    elif network_name == "VGG11":
        # Conv1
        layers_dict[0] = layers.Convolutional2DLayer(
            nFilters=64,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=inputshape,
            arithmetic=arithmetic,
        )
        layers_dict[1] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[0].outputshape, act="relu", arithmetic=arithmetic
        )
        layers_dict[2] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[1].outputshape,
            arithmetic=arithmetic,
        )

        # Conv2
        layers_dict[3] = layers.Convolutional2DLayer(
            nFilters=128,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[2].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[4] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[3].outputshape, act="relu", arithmetic=arithmetic
        )
        layers_dict[5] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[4].outputshape,
            arithmetic=arithmetic,
        )

        # Conv3
        layers_dict[6] = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[5].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[7] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[6].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv4
        layers_dict[8] = layers.Convolutional2DLayer(
            nFilters=256,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[7].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[9] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[8].outputshape, act="relu", arithmetic=arithmetic
        )
        layers_dict[10] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[9].outputshape,
            arithmetic=arithmetic,
        )

        # Conv5
        layers_dict[11] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[10].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[12] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[11].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv6
        layers_dict[13] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[12].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[14] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[13].outputshape, act="relu", arithmetic=arithmetic
        )
        layers_dict[15] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[14].outputshape,
            arithmetic=arithmetic,
        )

        # Conv7
        layers_dict[16] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[15].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[17] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[16].outputshape, act="relu", arithmetic=arithmetic
        )

        # Conv8
        layers_dict[18] = layers.Convolutional2DLayer(
            nFilters=512,
            kernelsize=3,
            stride=1,
            padding=1,
            act=None,
            bias=True,
            inputshape=layers_dict[17].outputshape,
            arithmetic=arithmetic,
        )
        layers_dict[19] = layers.BatchNormalizationLayer(
            inputshape=layers_dict[18].outputshape, act="relu", arithmetic=arithmetic
        )
        layers_dict[20] = layers.MaxPooling2DLayer(
            kernelsize=2,
            stride=2,
            padding=0,
            inputshape=layers_dict[19].outputshape,
            arithmetic=arithmetic,
        )

        # Classifier
        layers_dict[21] = layers.GlobalAveragePooling2DLayer(
            inputshape=layers_dict[20].outputshape, arithmetic=arithmetic
        )
        layers_dict[22] = layers.FullyConnectedLayer(
            n_out=4096,
            n_in=layers_dict[21].outputshape,
            act="relu",
            arithmetic=arithmetic,
        )
        layers_dict[23] = layers.DropoutLayer(
            p=0.5, inputshape=layers_dict[22].outputshape, arithmetic=arithmetic
        )
        layers_dict[24] = layers.FullyConnectedLayer(
            n_out=4096,
            n_in=layers_dict[23].outputshape,
            act="relu",
            arithmetic=arithmetic,
        )
        layers_dict[25] = layers.DropoutLayer(
            p=0.5, inputshape=layers_dict[24].outputshape, arithmetic=arithmetic
        )
        layers_dict[26] = layers.FullyConnectedLayerWithSoftmax(
            n_out=nClasses, n_in=layers_dict[25].outputshape, arithmetic=arithmetic
        )
    else:
        raise ValueError(
            f"Network architecure name {network_name} not recognized.")

    return layers_dict
