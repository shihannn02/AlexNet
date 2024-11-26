from torch import nn
from torch.nn import Module


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        # first layer, input: 32*32*3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # output: 32*32*64
        self.relu1 = nn.ReLU()
        # first pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # the output is: 16*16*64

        # second layer, input: 16*16*64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)  # output 16*16*192
        self.relu2 = nn.ReLU()
        # second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output is: 8*8*192

        # third layer, input is: 8*8*192
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)  # output: 8*8*384
        self.relu3 = nn.ReLU()

        # fourth layer, input: 8*8*384
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)  # output: 8*8*256
        self.relu4 = nn.ReLU()

        # # add one convolutional layer, input: 8*8*256
        # self.conv_add = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        # self.relu_add = nn.ReLU() # output: 8*8*384

        # fifth layer, input: 8*8*384
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)  # output: 8*8*384
        self.relu5 = nn.ReLU()
        # fifth pooling layer
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 4*4*384
        #

        # the sixth layer (fully connected layer), input: 4*4*384
        self.fc1 = nn.Linear(in_features=6144, out_features=6144)
        self.relu6 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)

        # the seventh layer (fc), input: 6144
        self.fc2 = nn.Linear(in_features=6144, out_features=6144)
        self.relu7 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)
        #
        # # # the add layer(fc), input: 6144
        # self.fc_add = nn.Linear(in_features=6144, out_features=6144)
        # self.relu_add = nn.ReLU()
        # self.drop_add = nn.Dropout(p=0.5)

        # the eighth layer (fc), input: 6144
        self.fc3 = nn.Linear(in_features=6144, out_features=10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        y = self.relu3(y)

        y = self.conv4(y)
        y = self.relu4(y)

        # y = self.conv_add(y)
        # y = self.relu_add(y)

        y = self.conv5(y)
        y = self.relu5(y)
        y = self.pool5(y)

        y = y.view(y.shape[0], -1)  # flatten

        y = self.fc1(y)
        y = self.relu6(y)
        y = self.drop1(y)

        y = self.fc2(y)
        y = self.relu7(y)
        y = self.drop2(y)

        # y = self.fc_add(y)
        # y = self.relu_add(y)
        # y = self.drop_add(y)

        y = self.fc3(y)

        return y