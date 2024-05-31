# Project Title : Convolutional Neural Networks for Image Classification on Fashion-MNIST Dataset

## Abstract
Recently, deep learning has been widely used in a wide range of industries. Among the deep neural network families, convolutional neural networks (CNNs) yield the most reliable outcomes when applied to real-world problems. Fashion brands have used CNN in their e-commerce to handle a range of problems, such as apparel recognition, search, and recommendation. A key component of each of these techniques is image categorization.  This work discusses the concept of classifying fashion-MNIST photos using convolutional neural networks. This research proposes a new CNN-based architecture to train CNN parameters using the Fashion MNIST dataset.  This study uses maxpooling, batch normalization, five convolutional layers, dropout and finally linked layers for classification. Various optimizers, learning rates, batch sizes and 100 epochs are used in the experiments. The outcomes displayed that the accuracy of the findings is influenced by the selection of the activation function, dropout rate and optimizer.
On the other hand, the fashion-MNIST dataset includes 28x28 grayscale pictures of 70,000 fashion items from 10 classifications, with 7,000 images in each class. The training set has 48,000 images, the evaluation set contains 6,000 images, and the test set contains 6,000 images. According to experimental results, the suggested model's accuracy exceeded 93%.


## Proposed Method
Five convolutional layer groups, five max-pooling layer groups, one fully connected layer, and a final softmax layer make up the new MCNN model. The activation function Relu, similar padding, stride one, a fixed kernel size of 3 by 3 (fixed), and the quantity of input and output channels in each convolutional layer make up the convolutional layer groups of our model. In addition, after every convolutional layer, a max-pooling layer is added, batch normalisation and RELU activation functions are applied, and Dropout is performed after every convolutional layer group. Then, a fully connected layer is included before softmax layers that flatten 2D spatial maps for image classification. 


## Dataset
Fashion MNIST dataset



## Results
  In this research, we apply a new model on the Fashion MNIST dataset. Five convolutional, max-pooling, batch normalisation, dropout, and fully-connected layer building blocks make up the suggested model. The suggested architecture is tested using the Fashion-MNIST dataset, which comprises of 28x28 scaled grayscale images of 10 classes with 48,000 training, 6,000 test, and 6,000 validation images.
To enhance the model’s performance, we employ various hyper parameter optimization techniques (varying the number of fully connected layers, batch size, stride, with or without dropout, etc.). We plan to conduct a comprehensive comparison between different pre-trained architectures (such as ResNet50 and VGG16). Proposed model gives a higher performance (an accuracy over 93% was obtained) as compared to other existing models.


## How to Use
You can upload 'Compuer_vision_fashion_mnist1.ipynb' in jupyter notebook or google colab.




## Project History
This project was originally completed in 2021. The commit history has been adjusted to reflect the original dates of the work.
