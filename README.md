# BeatNet

Predict the BPM of music using neural networks.

## Prerequisites

The simplest way to configure your environment is to install and use [Conda](https://conda.io/). If you have Conda installed you can create a working environment using either of the two following commands.

```bash
# GPU package for CUDA enabled devices
$ conda create -n <your_env_name> tensorflow-gpu

# Regular package (not recommended for training)
$ conda create -n <your_env_name> tensorflow
```

Alternatively you can install a tesorflow compatible version of Python, such as [Python 3.6](https://www.python.org/downloads/release/python-369/), then install tensorflow using pip.

```bash
$ pip install tensorflow
```

## Producing Training Data

See the [README](/datagen/) in the `datagen` directory for information on producing training data.

2000 sample training data images are included in this repo as an example. Extract [specgrams.zip](/specgrams.zip) in the root directory to use these.
    
## Training The Model

To train the model, simply run the `training.py` script with training data present in a directory named specgrams.

    $ python training.py <epochs>
    
This script takes an optional parameter of the number of epochs to perform. If left unspecified this defaults to 10.

## Using The Model

Once you have a trained model you can use the `predict.py` script to predict the BPM of a particular spectrogram image file like so:

    $ python predict.py <path_to_image.png>

This should print the models output to the console.
