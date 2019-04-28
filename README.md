# BeatNet

Predict the BPM of music using neural networks.

## Prerequisits

The following python packages are required and can be installed using the `pip install` command.

* `tensorflow==2.0.0-alpha0` - The Tensorflow library used for the ML backend
* `matplotlib` - Used to produce spectrograms of audio files

## Producing Training Data

The network trains over a set of .wav music files. In order to produce training data you must use [Rekordbox](https://rekordbox.com/) to beatgrid the tracks and place any tracks you wish to use for training in a playlist named `BEATNET`. Only .wav files are supported at present. You can export your library by going to `File > Export collection in xml format`. My library is included in the repo as an example although obviously I'm unable to distribute the music files due to copyright reasons, so this will need replacing with your own.

Assuming you have your library xml file located at `lib.xml` in the project root and a directory in the project root named `specgrams` you can produce training data by running the `import_lib.py` script as follows, where `XXX` is the number of samples to produce:

    $ python import_lib.py lib.xml XXX
    
## Training The Model

To train the model you can run the train script as follows:

    $ python training.py
    
*Note: this is not yet functional.*
