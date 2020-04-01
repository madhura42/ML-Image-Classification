# Image orientation classification 
The goal of this assignment is to classify the orientation of an image given a training dataset and testing dataset. The orientation of image can be classified as 0, 90, 180, 270  which are the 4 classes of our output variable.

## Data Description 
 The dataset contains: 
  photo id is a photo ID for the image.
  
  correct orientation is 0, 90, 180, or 270. 
  
  Features are 192, r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2 in the range 0-255.

Classifier:  K Nearest Neighbor
==========================
K nearest neighbor is a simple algorithm that looks at the k nearest or k most similar datapoints. The metric of similarity or distance can be computed by calculating the distance between the test datapoint and each sample of the training datapoint. Amongst these k nearest points, we check for the most commonly occurring output variable which in our case is the orientation. 
We experimented with two different distance/similarity measure: 
	Euclidean distance 
	Cosine similarity 

Training
./orient.py train train-data.txt knn_model.txt nearest


Testing 
./orient.py test test-data.txt knn_model.txt nearest

References : https://booking.ai/k-nearest-neighbours-from-slow-to-fast-thanks-to-maths-bec682357ccd

Classifier: Neural Network
==========================

In this we have implemented a fully-connected feed-forward neural network which
has following features: -

-   Input layer which has the same number of nodes as the size of the
    dimensional feature vector (ie. 8x8x3 = 192 in this case)

-   2 hidden layers of 128 nodes respectively

-   Output layer of the size of the classes (ie. 4 in this case)

**Approach:**

We have created two dictionaries initially which contains randomly generated
weights and biases for each layer. The output of each layer is calculated as:

Z = X\*weights + bias

Where X is the input to the layer and Z is the output of the layer.

Then network goes through a forward propagation where output to the first layer
is the input to the other layer and so on. While it is calculating the output of
each layer, it also keeps on updating the weights and biases. We also calculate
the cost after each feed forward propagation cycle. The error is calculated
using cross entropy. When feed forward propagation is done, it starts back
propagation. In back propagation, it starts from the output of the last layer
which is not our input. It keeps on moving to the first layer backwards and also
updates the weights and biases simultaneously. One feed forward propagation and
back propagation accounts for one iteration. The next iteration takes the
weights and biases updated in the previous iteration during the feed forward
propagation and so on.

After implementing the code, we tried lot of different combinations of alpha,
iterations, dropout probability, number of features of the hidden layers. We
found that the network works the best when we implement sigmoid activation
function for the first and hidden layers and softmax activation function for the
output layer. Also it works best when the hidden layers are 128 features
respectively. We also tried for different alpha and iterations as follows:-

| Alpha | Iterations | Training Accuracy | Testing Accuracy |
|-------|------------|-------------------|------------------|
| 0.01  | 500        | 56.23             | 54.86            |
| 0.1   | 1000       | 62.16             | 63.89            |
| 0.5   | 2000       | 67.69             | 62.14            |
| 1     | 2000       | 68.43             | 65.22            |
| 1     | 5000       | 70.84             | 67.44            |

**Design Decisions:**

-   We have used cross entropy cost (loss) function which is used to evaluate
    the model and diagnose how well the model is performing.
    
-   Training model are likely to overfit using few examples. Hence we have used
    dropout which drops the values which causes overfitting. Also to avoid
    overfitting, we have implemented L2 regularization.

-   We have used sigmoid as the activation function for the hidden two layers
    and softmax as the activation function for the output layer.

-   We have created different architectures so that it is easy to experiment
    with different training and testing samples. These architectures has flexibility of creating different number of hidden layers with different number of features.
   
references: 
1. Fashion product image classification using Neural Networks - https://towardsdatascience.com/fashion-product-image-classification-using-neural-networks-machine-learning-from-scratch-part-e9fda9e47661
2. Building a Neural Netwok from Scratch https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/


# Classifier: Decision Trees

Decision Trees are classification algorithms which are represented as binary trees where each node of the tree can have two child nodes. The idea is to split a labeled dataset such that it reduces the overall entropy/randomness of the data and makes it easier to classify new data. The node represents a point where the dataset is split.

We have used the metrics information entropy and information gain to determine what the best possible split is in a dataset. Information entropy is a measure of the randomness the data has. Consider a simple dataset with classes C = {a, b, c} and n elements. The information entropy is defined as:

       E = -[ pa*log2(pa) + pb*log2(pb) + pc*log2(pc) ]

Where pi is the proportion elements belonging to class i in the dataset (pi = ni/n). The entropy for a dataset with only 1 class would be 0 which means the data has no randomness.

Now we split the data at any arbitrary point into two branches, left and right. We can calculate the entropy of each branch as Eleft (with l elements) and Eright using the method described above. Entropy after splitting, Esplit can be determined by weighting the entropy of each branch by how many elements it has.

       Esplit = (nleft/n)*Eleft  + (nright/n)*Eright

The metric Information gain, which is the difference in the entropy of a dataset before and after splitting, measures how good a split is. It is given as:

	Gain = E - Esplit

The higher the information gain, the more the entropy removed. In other words, the information gain would be highest for a split which reduces the overall entropy. When we train the decision tree, we choose the best split as the one with the maximum information gain.

The accuracy of the decision tree depends on the depth of the tree. The higher the depth, the more complex our tree and the better we are able to separate the classes as efficiently as possible. Once all the branches in our tree are reduced to leaf nodes, we have successfully trained the decision tree.

References:

1. Random Forests for Complete Beginners - https://victorzhou.com/blog/intro-to-random-forests/
2. A Simple Explanation of Information Gain and Entropy - https://victorzhou.com/blog/information-gain/
