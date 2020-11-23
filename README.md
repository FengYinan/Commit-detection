# GAD
 The code of graph adnomal detection project

## Requeriment

Pytorch

Tensorflow (Tensorboard)

[pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)

## Code Structure

main.py contains parameters needed by the use. It is a good start if you want to get yourself familiar with code. DGAD.py contains trainning and testing process. The 3D graph convolution is defined in d3_graph_conv.py and net.py contains the whole network class. trans_graph.py is used to transform the .csv dataset to .npz gaph format.

The path to the input dataset can be changed in DGAD.py.

Lis_GAD.py and Lis_net.py contains the method in [previous work](http://www.public.asu.edu/~jundongl/paper/SDM19_DOMINANT.pdf).

## Resource

Dataset, related paper and our slide can be found on https://drive.google.com/drive/folders/1zXSLLmbTnJhLSYeSZ41N8ZNcGfH2VmLn?usp=sharing

Technical Reports is on https://www.overleaf.com/read/nddvkswmdrjr

## TO DO
Add ClusterGCN code

Add code comments
