# TFG- Modeling aerial and satellite data using NeRF

## Repository guide

The [src](src) folder contains the main code for the project.

The [src/dataloaders](src/dataloaders) contains the code for loading the datasets I have developed. The main are MultispectralSyntheticEODataloader for my custom datasets and the NerfDataloader for loading the original NeRF images. The others are other attempts for different data types.

The [src/models](src/models) contains code for the models I have attempted to create during the project, including special layers I have used.

The [src/training](src/training) contains all the boilerplate code I have developed to be able to train the models. This includes the main training loop, a Trainer class that does the training and some decorators to add functionalities to the training like  doing validation or loging to Tensorboard.

The [src/volume_render](src/volume_render) contains the main code for the rendering using NeRF and also the **main** code for the experiments.

The main entrypoints for the experiments are:
- [src/volume_render/render.py](src/volume_render/render.py) for the normal NeRF images,
- [src/volume_render/multispectral_rendering.py](src/volume_render/multispectral_rendering.py) for the multispectral normal NeRF images
- and [src/volume_render/multispectral_eo_rendering.py](src/volume_render/multispectral_eo_rendering.py) for the multispectral satellite NeRF images.


The [scripts](scripts) folder contains some scripts I have used on this project.

The most important script is [scripts/h2m.py](scripts/h2m.py) which I have used to generate the datasets.

