This code is a PyTorch implementation based on the article "EGAT- Edge-Featured Graph Attention Network" [EGAT: Edge-Featured Graph Attention Network | Artificial Neural Networks and Machine Learning – ICANN 2021 (acm.org)](https://dl.acm.org/doi/abs/10.1007/978-3-030-86362-3_21).

### Environment Requirements

- Python 3.10
- Pytorch 2.0.1
- DGL: [Deep Graph Library (dgl.ai)](https://www.dgl.ai/pages/start.html)

### Project Structure and Function Descriptions

#### layers.py – EGAT Model Layer Descriptions

- ```
  class MultiInputMLPUpdate(nn.Module) 
  ```

  - Implements a multi-input MLP (Multi-Layer Perceptron) update model class in PyTorch, used for updating edge features in graph neural networks.
  - `__init__` defines three input layers (corresponding to different input feature dimensions), a hidden layer, and an output layer, all transferred to a CUDA device for accelerated computation.
  - `update_edge` updates the features of each edge. It takes a dataset of edges (`edges`) as input, extracts the source node, target node, and edge features associated with each edge, transforms these features through corresponding linear layers, and sums the transformed results to obtain a combined feature vector `h`.
  - `forward` first prints the entire graph structure, transfers it to the GPU, then applies the `update_edge` function to update all edge features, and finally returns the updated edge features `graph.edata['e_out']`, which typically serve as input for loss function computation or further graph neural network operations during training or inference.

- ```
  class EGATLayer(nn.Module)
  ```

  - This class is a custom layer based on the Graph Attention Network (GAT), designed to handle node and edge features. During initialization, it sets up different linear layers and parameter matrices and initializes these parameters.
  - In the `forward` function, it computes attention weights based on node and edge features. Node information is updated by applying a message-passing function that considers neighbor nodes and edge information, using the Leaky ReLU activation function.
  - The updated node features are linearly transformed again to obtain node-related attention weights, which are combined with the original edge features to compute new edge attention weights.
  - The updated node and edge features are combined and passed into a multi-input MLP class named `MultiInputMLPUpdate` to further update the edge features.
  - Finally, the MLP-updated edge features are added to an optional bias term to produce the final edge feature output.

- ```
  class BottleneckLayer(nn.Module)
  ```

  - Simultaneously applies linear transformations to node and edge features, serving as an information bottleneck or feature compression intermediate layer.
  - It performs dimensionality reduction or expansion on node and edge features through fully connected layers, followed by nonlinear transformation using the ELU activation function.

- ```
  class MLPPredictor(nn.Module)
  ```

  - Defines a simple Multi-Layer Perceptron (MLP) prediction model for predicting outputs from input data.
  - Typically used in the final layer of the network to transform processed features into output class probability distributions.

- ```
  class MergeLayer(nn.Module)
  ```

  - Merges feature vectors from different sources or after multi-stage processing. For example, when there are multiple feature streams in a stage, this layer can fuse these features into a new, lower-dimensional feature vector through fully connected layers.

#### models.py – Integration of Layers

- The `EGAT` class defines a custom model based on GAT (Graph Attention Network), combining components such as bottleneck layers, multi-input GAT layers, MLP predictors, and merge layers.
- In the initialization method `__init__`:
  - Parameters are defined: `node_feats` (node feature count), `edge_feats` (edge feature count), `f_h` and `f_e` (hidden layer feature counts), `lamda` (lambda parameter), `num_heads` (number of attention heads), `dropout` (dropout rate), `pred_hid` (prediction layer hidden layer size), and `l_num` (number of network layers).
  - A bottleneck layer object `bottleneck` is constructed to perform initial linear transformation and nonlinear activation on input node and edge features.
  - A custom `EGATLayer` object `egat` is built to implement the graph attention mechanism for updating node and edge features.
  - An `MLPPredictor` object `pred` is constructed for final classification or regression prediction tasks.
  - A `MergeLayer` object `merge` is built to fuse multi-layer features into a single layer.


#### utils.py – Dataset Creation and Result Evaluation

- `encode_onehot(labels)` function:
  - Input: A list of labels `labels`, where each element represents a category label.
  - Output: Converts the input discrete category labels into a one-hot encoded 2D array.
- `normalize(mx)` function:
  - Input: A sparse matrix `mx` (assumed to be of type scipy.sparse).
  - Output: A sparse matrix normalized row-wise.
- `accuracy(output, labels)` function:
  - Input parameters: `output` is the model's prediction results (typically a vector containing probabilities for each class), and `labels` are the actual labels.
  - First, the class index with the highest probability for each sample is found from the output and converted to the same type as the labels. The number of correctly predicted elements is then calculated by comparing predicted classes with true labels and converting to a floating-point boolean value, followed by summation. Finally, the number of correct predictions is divided by the total number of samples to obtain the accuracy.
- `load_data()`:
  - This function generates a simulated dataset and preprocesses the data.
  - First, a graph structure with 8 nodes and 30 edges is randomly generated, with 20-dimensional features assigned to nodes and edges.
  - For labels, the `encode_onehot` function encodes discrete labels into one-hot encoding, with an example label list provided.
  - Node and edge features are then normalized. The dataset is divided into training, validation, and test sets, with indices recorded in `idx_train`, `idx_val`, and `idx_test`, respectively.
  - All data is converted to PyTorch Tensor type for computation within the PyTorch framework.
  - Finally, the entire graph object, node features, edge features, labels, and the partitioned training, validation, and test set indices are returned.