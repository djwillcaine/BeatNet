# BeatNet

Predict the BPM of music using neural networks.

## Prerequisites

The following python packages are required and can be installed using the `pip install` command.

* `tensorflow==2.0.0-alpha0` - The Tensorflow library used for the ML backend
* `matplotlib` - Used to produce spectrograms of audio files

## Producing Training Data

**Note:** Included with the project is a zip file containing 1000 training data sample images. Extract this file in the project root to begin training right away using this training data.

The network trains over a set of png spectrograms derrived from .wav music files. In order to produce training data you should use [Rekordbox](https://rekordbox.com/) to beatgrid the tracks and place any tracks you wish to use for training in a playlist named `BEATNET`. Only .wav files with a single BPM marker are supported at present. You can export your library by going to `File > Export collection in xml format`. My library is included in the repo as an example although obviously I'm unable to distribute the music files due to copyright reasons, so this will need replacing with your own.

Assuming you have your library xml file located at `lib.xml` in the project root you can produce training data by running the `import_lib.py` script as follows, where `XXX` is an optional parameter for the number of samples to produce:

    $ python import_lib.py lib.xml XXX
	

    
## Training The Model

To train the model you can run the train script as follows:

    $ python training_bpm_only.py XXX model.h5
    
Here, `XXX` is the number of epochs to train for and `model.h5` is the filename that the model will be saved as in the `temp` directory.

## Using The Model

Once you have a trained model you can use the `predict.py` script to predict the BPM of a particular spectrogram image file like so:

    $ python predict.py temp/model.h5 test_image.png

This should print the models output to the console.

## License

This project is not licensed. All rights reserved.
