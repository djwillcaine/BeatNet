# BeatNet

Tempo estimation of electronic music using CNNs trained on a DJ's own music library.

This project was undertaken for my dissertation in BSc Hons. Computer Science at Newcastle University. The resulting models are not well performing, but the source code is available here under the GNU General Public License v3.0 for anyone to use. You may have better luck than me by training models on your own library of music, or perhaps the [GiantSteps Tempo Set](https://github.com/GiantSteps/giantsteps-tempo-dataset).

### Project Results

You can view evaluated results of the included models [here](eval/results.csv). Additionally, a verbose log of predictions for the test set for each model can be found in the [eval](eval/) directory. As can be seen from these results, none of the models trained were accurate enough to be practically useful.

### Ackownledgments

A big thank you goes to **Dr. Jaume Bacardit** (@jaumebp) for being my project supervisor.

Additionally, I'd like to thank [Decizion](https://soundcloud.com/decizionmusic) and [Aperio](https://soundcloud.com/aperio) for their support.

## Prerequisites

The simplest way to configure your environment is to install and use [Conda](https://conda.io/). If you have Conda installed you can create a working environment using either of the two following commands.

```bash
$ conda create -n <your_env_name> tensorflow
```

Alternatively you can manually [install TensorFlow](https://www.tensorflow.org/install/pip?lang=python3) for Python, then install the other requirements using pip as shown below.

```bash
$ pip install -r requirements.txt
```

## Producing Training Data

In order to produce training data it is recommended that you use [Rekordbox](https://rekordbox.com/) to annotate your music library as accurately as possible. **Be mindful of tracks that change BPM part way through**, as these may result in innacturate training data. Place all tracks you wish to use for data generation in a playlist named `BEATNET`. Then, export your library by clicking **File** > **Export Collection in xml format** and saving the file to the root directory of this project as `lib.xml`.

Once you have done this, you are able to generate training data using the `generate.py` script as shown below. You can expect to generate approximately 10x as many images as the number of tracks specified.

```bash
$ python generate.py -n 1000
```

### Density Plots for Training Data

A utility script is provided for visualising the density plots of 1 or more datasets. To use this script, each dataset (even if only using one) must be placed in it's own subdirectory in the `data` directory.

```bash
$ python density_plot.py
```
    
## Training a Model

Once you have some training and validation data, you can train the model using the `train.py` script. For a comprehensive list of options specify the `--help` flag. The most commonly needed options are as follows, though:

- `--architecture [deep|shallow]` - Produce either a deep or a shallow model architecture. Defaults to *shallow*.
- `--output-mode [classification|regression]` - Produce either a classification or a regression type model. Defaults to *classification*.
- `--model-name model_name` - Specify an optional model name. It is recommended to include the dataset BPM range, in the format `min-max`, somewhere in this name. For example: *model.80-180*. Defaults to *data_set.architecture.output_mode.bpm_range*.
- `--data-dir dir` - Specify the data directory to use for training. The directory should contain a *training* and a *validation* subdirectory. Defaults to *data*.
- `--epochs E` - Specify the maximum number of epochs to train for, *E*. Defaults to *100*.
- `--steps-per-epoch S` - Specify the number of steps, *S*, per epoch. Defaults to *100*.
- `--batch-size B` - Specify the batch size, *B*, to train with. A lower batch size can help reduce crashes. Defaults to *128*.

```bash
$ python train.py --architecture deep --output-mode classification
```

## Evaluating The Model

To evaluate a model, you need a separate test dataset. This will be automatically generated by `generate.py` as long as you did not specify a test-split of 0. This script takes an optional `-w` flag that will write the evaluation results to file. Model metrics will be written to `eval/results.csv` and a verbose log of predictions will be written to `eval/model_name.csv`. If the flag is not specified, the metrics will simply be printed to the console.

```bash
$ python evaluate.py path_to_model.h5
```

## Batch Training/Evaluating

In order to train and evaluate a batch of all possible models given multiple datasets, the `train_eval.py` utility script is provided. You must place each dataset (even if only using 1) in their own subdirectory of the `data` directory. For example `data/DS1`, `data/DS2`, etc... The script can then be invoked like so:

```bash
$ python train_eval.py
```

## Using a Model

You can use a trained model to estimate the BPM of either an audio file or a pre-generated spectrogram. You must specify a model path as well as either an image or an audio file. Also be careful to use the same sample length as was used to generate the training data, if specifying an audio file.

- `--model-path path_to_model.h5` - Used to specify the file path to the model to use for estimations. Defaults to *model.h5*.
- `--track-path path_to_audio.mp3` - Specify the audio file to estimate the BPM for. 
- `--sample-length S` - Use a sample length of *S* seconds. Must match the sample length used to train the model. Defaults to *10*.
- `--image-path path_to_specgram.png` - Optionally used to specify the file path to a pre-generated spectrogram, instead of an audio file.

```bash
$ python estimate.py --model-path models/DS1.deep.classification.80-180.best.h5 --track-path path_to_audio.mp3
```

## License

Tempo estimation of electronic music using CNNs trained on a DJ's own music library.  
Copyright (C) 2020 Will Caine

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.