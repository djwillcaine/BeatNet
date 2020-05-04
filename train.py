import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Concatenate, Input, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import os
import datetime
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 40

def train_model(batch_size, steps_per_epoch, epochs, data_dir, model_name, output_mode, depth):

    # Define variables
    train_dir = os.path.join(data_dir, 'training')
    validation_dir = os.path.join(data_dir, 'validation')

    classes = [int(x) for x in os.listdir(train_dir)]
    n_classes = len(classes)

    if model_name == None:
        model_name = '%s.%s.%s.%d-%d' % (
            os.path.basename(data_dir),
            depth,
            output_mode,
            min(classes),
            max(classes))

    # Load datasets
    is_categorical = output_mode == 'classification'
    train_data = fetch_dataset(train_dir, batch_size, is_categorical, repeat=True)
    validation_data = fetch_dataset(validation_dir, batch_size, is_categorical)

    # Configure output layer
    if is_categorical:
        output_layer = Dense(n_classes, activation='softmax', name='Output')
    else:
        output_layer = Dense(1, activation='elu', name='Output')

    # Configure model architecture
    if depth == 'deep':
        model = build_deep_model(output_layer, model_name)
    elif depth == 'shallow':
        model = build_shallow_model(output_layer, model_name)
    else:
        print('Unknown depth given: %s' % output_mode)
        print('Please specify one of: deep, shallow')
        return

    # Compile model
    if is_categorical:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print(model.summary())

    # Callback to log data for TensorBoard
    log_dir = "temp\\logs\\fit\\" + model_name
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Callback to save best weights
    checkpoint = ModelCheckpoint('models/%s.best.h5' % model_name, verbose=1, save_best_only=True)

    # Callback for early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    # Train model
    history = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[tensorboard, checkpoint, early_stop]
    )

    # Save final model
    model.save('models/' + model_name + '.final.h5')

    # Plot graphs
    plot_graph(model_name, history, is_categorical)


def img_to_tensor(img_path):
    img_raw = tf.io.read_file(img_path)
    image = tf.image.decode_png(img_raw, channels=1)
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
    image /= 255.0
    return image


def fetch_dataset(ds_dir, batch_size, is_categorical, repeat=False):
    if not os.path.isdir(ds_dir):
        print("Dataset directory '%s' not found" % ds_dir)
        return

    image_paths = []
    image_labels = []

    # Iterate over dataset directories
    for subdir in pathlib.Path(ds_dir).iterdir():
        label = int(subdir.stem)
        for file in subdir.iterdir():

            if file.suffix != ".png":
                continue

            image_paths.append(os.path.abspath(str(file)))
            image_labels.append(label)

    # Encode labels if in categorical mode
    if is_categorical:
        m = min(image_labels)
        image_labels = [i - m for i in image_labels]
        image_labels = tf.keras.utils.to_categorical(image_labels)

    # Build dataset from paths/labels
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(img_to_tensor)
    label_ds = tf.data.Dataset.from_tensor_slices(image_labels)
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.shuffle(buffer_size=len(os.listdir(ds_dir)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1)

    if repeat:
        ds = ds.repeat()

    return ds


def build_shallow_model(output_layer, model_name):
    return Sequential([
        # Short filters
        Conv2D(16, (5, 1), padding='same', activation='elu', name='Conv1.1', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        Conv2D(16, (5, 1), padding='same', activation='elu', name='Conv1.2'),
        BatchNormalization(name='BN1'),
        AveragePooling2D(pool_size=(1, IMG_HEIGHT), name='AvgPool'),
        
        # Long filters
        Conv2D(16, (IMG_WIDTH, 1), padding='same', activation='elu', name='Conv2.1'),
        Conv2D(16, (IMG_WIDTH, 1), padding='same', activation='elu', name='Conv2.2'),
        BatchNormalization(name='BN2'),

        Conv2D(32, (1, 1), padding='same', activation='elu', name='1x1'),

        # Dense layers
        GlobalAveragePooling2D(name='GlobalAvgPool'),
        Dense(64, activation='elu', name='FC'),
        BatchNormalization(name='BN4'),
        output_layer
    ], name=model_name)


def build_deep_model(output_layer, model_name):
    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1))

    # Short filters
    x = BatchNormalization(name='BN1')(inputs)
    x = Conv2D(16, (5, 1), padding='same', activation='elu', name='Conv1')(x)
    x = BatchNormalization(name='BN2')(x)
    x = Conv2D(16, (5, 1), padding='same', activation='elu', name='Conv2')(x)
    x = BatchNormalization(name='BN3')(x)
    x = Conv2D(16, (5, 1), padding='same', activation='elu', name='Conv3')(x)

    # Multi-filter modules
    pools = [(1, 5), (1, 2), (1, 2), (1, 2)]
    filters = [32, 64, 96, 128, 192, 256]

    for i, pool_size in enumerate(pools):
        x = AveragePooling2D(pool_size=pool_size, name='mf%d_AvgPool' % i)(x)
        x = BatchNormalization(name='mf%d_BN' % i)(x)

        convs = [Conv2D(24, (f, 1), padding='same', name='mf%d_Conv_%dx1' % (i, f))(x) for f in filters]

        x = Concatenate(name='mf%d_Concat' % i)(convs)
        x = Conv2D(36, (1, 1), name='mf%d_Conv_1x1' % i)(x)

    # Dense layers
    x = BatchNormalization(name='BN4')(x)
    x = Dropout(0.5, name='DO')(x)
    x = Flatten(name='Flat')(x)
    x = Dense(64, activation='elu', name='FC1')(x)
    x = BatchNormalization(name='BN5')(x)
    x = Dense(64, activation='elu', name='FC2')(x)
    x = BatchNormalization(name='BN6')(x)
    x = output_layer(x)

    return Model(inputs=inputs, outputs=x, name=model_name)

def plot_graph(model_name, history, is_categorical):
    # Taken and modified from: https://www.tensorflow.org/tutorials/images/classification
    if is_categorical:
        metric = history.history['accuracy']
        val_metric = history.history['val_accuracy']
        metric_label = 'Accuracy'
    else:
        metric = history.history['mae']
        val_metric = history.history['val_mae']
        metric_label = 'MAE'

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = [i + 1 for i in range(len(loss))]

    plt.figure(figsize=(8, 4))
    plt.suptitle(model_name)

    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.plot(epochs_range, metric, label='Training ' + metric_label)
    plt.plot(epochs_range, val_metric, label='Validation ' + metric_label)
    plt.legend()
    plt.title('Training and Validation ' + metric_label)

    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.savefig('graphs/' + model_name + '.png')


def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        # directory already exists
        pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-s', '--steps-per-epoch', type=int, default=100)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-d', '--data-dir', default='data')
    parser.add_argument('-n', '--model-name')
    parser.add_argument('-o', '--output-mode', choices=['classification', 'regression'], default='classification')
    parser.add_argument('-a', '--architecture', choices=['deep', 'shallow'], default='shallow')

    args = parser.parse_args()

    create_dir('models')
    create_dir('graphs')

    train_model(
        args.batch_size,
        args.steps_per_epoch,
        args.epochs,
        os.path.abspath(args.data_dir),
        args.model_name,
        args.output_mode,
        args.architecture)