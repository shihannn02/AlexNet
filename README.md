## AlexNet Implementation

There are totally three .py in the code file, with the environment of pytorch

### The function of .py files

`AlexNet_ori.py`: the original architecture of AlexNet, which input is 227*227*3. When using this architecture, we need to do some change to the image size of CIFAR10.

`AlexNet.py`: I change the parameters of AlexNet, so with this AlexNet architecture, we can feed into the pictures with size 32*32*3 directly.

`Train_AlexNet.py`: this .py file is used to load data and train the AlexNet.


### Usage of .py files

Open both AlexNet.py and train_AlexNet.py files, and you can run it directly with optimizer sgd, batch size=64, epoch=50, learning rate=0.005.

I used the optimizer of sgd in the train_AlexNet.py, if you want to change the optimizer, uncomment the rows below, for the optimizer of Adam.
