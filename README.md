# CS484 Final Project: Semi-Supervised Learning on CIFAR-10

## Team Members and Contributions:

| Name                | Email              | Student Number | Contributions: |
|---------------------|--------------------|----------------|----------------|
| Stone Hu            | ds2hu@uwaterloo.ca | 20890769       | Research/implementation for loss functions, model development, explanation of loss functions and gathered conclusions |
| Justin Metivier     | jmetivie@uwaterloo.ca | 20874949    | Development of generic model for loss function development and MNIST testing, custom data loader for semi-labeled data, model training, helped retrofit models to have specified loss functions |
| Andrew Batek        | abatek@uwaterloo.ca | 20892302      | Preliminary paper review, LeNet implementation, hyperparameter tuning and model training, training result aggregation |
| Matthew Erxleben    | merxlebe@uwaterloo.ca | 20889980    | Preliminary research on model architecture, ResNet implementation, jupyter notebook tables/graphs/notes |

## Code Libraries:

- PyTorch: Model construction and training
- PyTorch Lightning: Data loading tools
- Torchvision: Datasets sourcing
- TensorBoard: Tracking training and visualizing performance

## Project Topic 5: Semi-Supervised Image Classification

## Abstract:
The overarching theme of this project is the implementation of semi-supervised learning across different CNN architectures with different semi-supervised losses.
There were two main goals with this: 1 - to understand implementation and effectiveness of different semi-supervised techniques and loss functions, and
2 - to compare the robustness in prediction accuracy of different models across different ratios of labeled to unlabeled data. 
We focused on 2 different model architectures - an implementation of LeNet and an implementation of ResNet.
They were modified to use a traditional supervised loss function of cross entropy in conjunction with different unsupervised loss techniques to handle the presence of labeled and unlabeled data simultaneously.
These loss techniques are mutual information loss, entropy minimization loss, consistency regularization loss, and K-Means clustering loss.

This project compares the accuracy of LeNet and ResNet using all four of these loss functions on the CIFAR-10 Dataset.

## Project Overview:
We implemented 3 different models. The first is a generic CNN architecture which we implemented with cross entropy loss as the supervised loss and a variety of different unsupervised loss techniques.
We tested Mutual Information loss, sourced from Bridle & MacKay "Unsupervised Classifiers, Mutual Information and Phantom Targets", NIPS 1991, Entropy Minimization loss, and Consistency Regularization loss,
and chose the best performing of the three for our final model architecture. This model was used as more of a testing ground to develop the different loss functions and do preliminary research on effectiveness of different losses.
We then chose two preexisting CNN architectures to modify using our findings from the first model.
The first model is an implementation of LeNet (Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, Nov. 1998).
Our second model is an implementation of ResNet (He, Kaiming, X. Zhang, Shaoqing Ren and Jian Sun. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition).
In addition to the three aforementioned loss functions, we also implemented a K-means loss function using the deep features of the labeled data for these two models.
Our project compares the effectiveness of the different loss functions on these two models, as well as compares robustness across different ratios of labeled to unlabeled data.

We initially chose MNIST as our dataset as it was easier to develop around, however, after finding good results across all models,
we then chose to modify each to be trained on CIFAR-10, as we hoped this would accentuate any differences between models, loss functions, and label ratio.

## Models:

#### Baseline CNN:
This model aimed to implement a simple, no-frills CNN architecture to see if there were any advantages to a simpler model while handling a mix of labelled and unlabeled data.
It is 2 convolutional layers followed by a linear layer. We also used it as a sandbox to implement the different loss functions and ensure they were performing as expected.
It performed quite well on MNIST, however, we decided not to pursue comparing it with CIFAR-10 as the other two models were overall better. We also knew the limitations of the
depth of the architecture when dealing with RGB images would lead to subpar results.

#### LeNet:
We employed LeNet, one of the first convolutional neural network architectures introduced by Yann LeCun in 1998. With only 5 layers, LeNet is relatively shallow and more suited
for grayscale images. Its simplicity makes it less suited for larger, RGB input, but it is a good demonstration of early convolutional neural networks.

#### ResNet:
We also implemented ResNet, a popular neural network architecture introduced in 2015 that has become the backbone for many modern vision models. ResNet’s key innovation is the use of
residual/skip connections, which allow gradients to flow through many more layers without vanishing. In our project, we utilized on ResNet18, the smallest variant featuring 18 layers,
chosen to balance model capacity with our computational constraints. Due to the skip connections, ResNet18 handles larger RGB images well and allows for exploring deeper versions of ResNet in future work.

## Loss Functions:
For all our models, we used the standard cross-entropy loss as our loss function for labeled (supervised) data, and augmented this loss with a variety of unsupervised loss functions,
comparing the effectiveness of each by using each one in isolation with our supervised loss. Our 4 loss functions were KMeans clustering loss, mutual information loss, entropy minimization loss,
consistency regularization loss. Each of these loss functions is motivated by an assumption that we make about our dataset.

In KMeans clustering loss, we add an additional set of parameters cluster_centers to each model. The idea behind this is that when we extract the deep features of an image using a CNN,
hopefully the points representing the deep features for data samples of the same class (i.e. two images of airplanes) end up close to each other. With this idea, we assume that the deep
features for inputs of the same class will naturally end up in clusters, which we will model using KMeans clusters. We set the number of clusters to be the number of output classes that
the network is trying to predict, with the dimension of a point being the dimension of a deep feature vector for the neural net. Then, during training, we calculate the KMeans loss between
each deep feature and its closest cluster, and backpropagate this loss through both the convolutional network and the KMeans cluster center parameters to update the weights of the network. 

In entropy minimization loss, we make no modifications to our network, but directly calculate an additional loss for unlabeled data in the training loop. The idea behind entropy minimization
is to directly use the formula for entropy as a loss function, and minimize it to reduce uncertainty in the network’s output probability vectors (i.e. make them closer to one-hot). The motivation
for this is that elements in our dataset have a clear correct class, and that we want to encourage our model to make more confident (low-entropy) classifications. When combined with labeled
training data, the network is more likely to make correct guesses, and entropy minimization loss pushes the network to assign higher probabilities to one single output class (which is hopefully
the correct one). To implement this, we take the entropy of the softmax probability vector from the network and add it to our cross entropy loss, and backpropagate this through the entire network.

In mutual information loss, we make no direct modifications to our network, but extract the deep features of the CNN and measure the correlation between those and the input features. The motivation
for mutual information loss is that there should be a strong correlation (i.e. high shared information) between the predicted class and the input image, so we formulate a loss function for the output
softmax probability vector based on this assumption. For our implementation, we expanded on this loss by instead extracting the deep features of the CNN instead, since we believed there would be a
stronger correlation between these and the input images. Then, we backpropagate the losses through the network.

For consistency regularization loss, we implement two forward passes through the network, one with the original image, and one with some noisy transformations applied to the image. We calculate a
loss based on the mean squared error between the two softmax probability outputs, and backpropagate this through the network. The motivation for this loss is that the predicted class for input images
shouldn’t change based on certain types of added noise, such as transformations, horizontal/vertical flips, etc., so we encourage the network to predict the same class for the original and noisy
versions of the same image.
