## AlexNet Implementation

There are totally three .py in the code file, with the environment of pytorch

### The function of .py files

`AlexNet_ori.py`: the original architecture of AlexNet, which input is 227 * 227 * 3. When using this architecture, we need to do some change to the image size of CIFAR10.

`AlexNet.py`: I change the parameters of AlexNet, so with this AlexNet architecture, we can feed into the pictures with size 32 * 32 * 3 directly.

`Train_AlexNet.py`: this .py file is used to load data and train the AlexNet.


### Usage of .py files

You can run the AlexNet directly on `Train_AlexNet.py` with the default configuration: SGD optimizer, batch size of 64, 50 training epochs, and a learning rate of 0.005. If you want to switch optimizers, Adam optimizer code is pre-configured in train_AlexNet.py - simply uncomment the relevant lines to use Adam instead of the default SGD optimizer.
