# Neural-network-classification-of-MNIST
The MNIST training data should be downloaded from http://yann.lecun.com/exdb/mnist/ and named train.csv in order to run the code. The dataset is too big to be hosted here.

In it's current form the classification accuracy is 95%. That's ok for the simple model of 30 hidden nodes, although quite bad compared to the state of the art convolutional networks or deeper networks. The backpropagation algorithm was simple gradient descent with momentum. The number of nodes and other parameters can be changed in the code.

By adding one more layer or more hidden nodes etc the accuracy could be improved. One of the purposes of making this code was to remind myself of c++ coding. The code is not completely polished, it has older code lines commented as I was playing around with dynamical allocating and had problems with stack overflow etc. A better and easier way atleast for me would be to use python and Keras for the classification. I will later add that to Github.
