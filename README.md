# Analysis of Deep Perceptual Similarity metrics
This repository contains code for evaluating and analyzing deep perceptual similarity.
It has three feature extraction networks available (AlexNet, SqueezeNet, and VGG-16) and five different methods for calculating deep perceptual similarity (compare features by spatial position, ordered by activation strength, averaged over channels, or the sum of spatial and any of the other two).
There are some additional options available, like whether to use channel-wise normalization of the network features.

## Requirements
The code has been tested and is working with Python 3.9, PyTorch 1.10.2, Torchvision 0.11.3, and PyTorch-Lightning 1.5.10

## Contents and Execution
The repo contains test images (images/) for analyzing different networks and calculations methods of deep perceptual similarity as well as code for running such analysis (analysis.py) and visualizing feature maps (feature_map.py).
Additionally, there is code for collecting the BAPPS dataset for perceptual similarity and evaluating the varuous networks and calculation methods on that dataset (experiment.py).

The analysis.py file will automatically perform the analysis and store the results in a logs folder if executed.
Both feature_map.py and experiment.py can take additional input arguments and flags. Run them with the --help flag to get a list of all options.
The experiment.py file will automatically collect the BAPPS dataset if it hasn't already been downloaded.

dataset_collector.py is only for collecting the BAPPS dataset and is automatically used by experiment.py
loss_networks.py is used for collecting various pretrained Torchvision networks and is imported by the files that need it.
