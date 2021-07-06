# Amazon-Community-Detection
Analyzed a graph dataset of books on United States politics sold through Amazon.com, where the nodes represent books and edges represent items that were co-purchased together. 

In this project, I performed semi-supervised learning using a Graph Convolutional Network (GCN) with Tensorflow. The constructed GCN had 3 layers, each of which used a hyperbolic tangent activation function. Due to the data having more than 2 classes, I opted to use a cross-entropy loss function, which could be easily adapted for more than 2 classes. The model was then trained over 300 epochs. 

Sources/Insipration:
https://github.com/tkipf/gcn
