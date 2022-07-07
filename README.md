# Analysis of Deep Perceptual Similarity metrics
This repository contains the code for evaluating and analyzing deep perceptual similarity that was used in the work ["Identifying and Mitigating Flaws of Deep Perceptual Similarity Metrics"](https://arxiv.org/abs/2207.02512).
It has three feature extraction networks available (AlexNet, SqueezeNet, and VGG-16) and five different methods for calculating deep perceptual similarity (compare features by spatial position, ordered by activation strength, averaged over channels, and the sum of spatial and any of the other two).
There are some additional options available, like whether to use channel-wise normalization of the network features.

As part of that it contains an implementation of the paper ["The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) and code for downloading the BAPPS dataset introduced in that paper.
While the LPIPS training from that paper is implemented, it is not currently guaranteed to be correctly.
Everything else has been verified to generate the same results.

## Requirements
The code has been tested and is working with Python 3.9, PyTorch 1.10.2, Torchvision 0.11.3, and PyTorch-Lightning 1.5.10

The experiment.py file currently uses Wandblogger which requires a "Weights and Biases"-account.
If you want to run experiment.py either setup WandB or change to another logger.

experiment.py will also be very slow running on CPU, especially if you run training as well.
The other content should run fine on CPU only.

## Contents and Execution
The repo contains test images (images/) for analyzing different networks and calculations methods of deep perceptual similarity as well as code for running such analysis (analysis.py) and visualizing feature maps (feature_map.py).
Additionally, there is code for collecting the BAPPS dataset for perceptual similarity and evaluating the varuous networks and calculation methods on that dataset (experiment.py).

The analysis.py file will automatically perform the analysis and store the results in a logs folder if executed.
Both feature_map.py and experiment.py can take additional input arguments and flags. Run them with the --help flag to get a list of all options.
The experiment.py file will automatically collect the BAPPS dataset if it hasn't already been downloaded.

dataset_collector.py is only for collecting the BAPPS dataset and is automatically used by experiment.py
loss_networks.py is used for collecting various pretrained Torchvision networks and is imported by the files that need it.

## Referencing
If you make a scientific work building on this repository please cite [the paper](https://arxiv.org/abs/2207.02512). BibTex provided below.
```
@article{sjogren2022identifying,
  title={Identifying and Mitigating Flaws of Deep Perceptual Similarity Metrics},
  author={Sj{\"o}gren, Oskar and Pihlgren, Gustav Grund and Sandin, Fredrik and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2207.02512},
  year={2022}
}
```

If you use BAPPS or the LPIPS implementation you should additionally cite ["The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html). BibTex below.
```
@inproceedings{zhang2018unreasonable,
  author = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  title = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
} 
```
