from torch import nn
from torch.nn import Module


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        # first layer, input: 227*227*3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4,padding=0)  # output: 55*55*96
        self.relu1 = nn.ReLU()
        # first pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # the output is: 27*27*96

        # second layer, input: 27*27*96
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)  # output 27*27*256
        self.relu2 = nn.ReLU()
        # second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # output is: 13*13*256

        # third layer, input is: 13*13*256
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)  # output: 13*13*384
        self.relu3 = nn.ReLU()

        # fourth layer, input: 13*13*384
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)  # output: 13*13*384
        self.relu4 = nn.ReLU()

        # fifth layer, input: 13*13*256
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)  # output: 13*13*256
        self.relu5 = nn.ReLU()
        # fifth pooling layer
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)  # output: 6*6*256
        #

        # the sixth layer (fully connected layer), input: 6*6*256=9216
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.relu6 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)

        # the seventh layer (fc), input: 4096
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)

        # the eighth layer (fc), input: 4096
        self.fc3 = nn.Linear(in_features=4096, out_features=10)

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

        y = self.fc3(y)

        return y
