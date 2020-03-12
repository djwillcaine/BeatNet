import tensorflow as tf
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
import sys
import pathlib

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64

class BeatNet:
    def __init__(self):
        self.model = None


    def load_model(self, model_path="model.h5"):
        if not os.path.isfile(model_path):
            print("Model file not found")
            return
            
        self.model = tf.keras.models.load_model(model_path)


    def gen_model(self):
        self.model = tf.keras.models.Sequential([
            # Convolution 1
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Flatten
            tf.keras.layers.Flatten(),

            # Full connection
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mae'])


    def img_to_tensor(self, img_path):
        img_raw = tf.io.read_file(img_path)
        image = tf.image.decode_png(img_raw, channels=1)
        image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
        image /= 255.0  # Normalise values
        return image


    def fetch_dataset(self, ds_dir, batch_size=512, repeat=False):
        if not os.path.isdir(ds_dir):
            print("Dataset directory '%s' not found" % ds_dir)
            return

        image_paths = []
        image_labels = []

        # Iterate over dataset directory
        files = pathlib.Path(ds_dir).iterdir()
        for filename in files:
            filename = str(filename)

            if filename[-4:].lower() != ".png":
                continue

            label = float(filename[:-4].split('-')[2])
            image_paths.append(os.path.abspath(filename))
            image_labels.append(label)

        # Build dataset from paths/labels
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        image_ds = path_ds.map(self.img_to_tensor)
        label_ds = tf.data.Dataset.from_tensor_slices(image_labels)
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        ds = ds.shuffle(buffer_size=len(os.listdir(ds_dir)))

        if repeat:
            ds = ds.repeat()

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=1)

        return ds


    def train(self, epochs=5):
        if (self.model == None):
            print("Model not yet instantiated, try beatnet.gen_model() first")

        # Load training and validation datasets
        training_data = self.fetch_dataset("data/training", repeat=True)
        validation_data = self.fetch_dataset("data/validation", batch_size=128)

        # Train model
        self.model.fit(training_data, epochs=int(epochs), steps_per_epoch=512, validation_data=validation_data)


    def predict(self, filename):
        if (self.model == None):
            print("Model not yet instantiated, try beatnet.load_model('model.h5') first")

        # Load image and label
        image = self.img_to_tensor(filename)
        image = tf.reshape(image, (1, IMAGE_WIDTH, IMAGE_HEIGHT, 1))
        label = float(os.path.basename(filename)[:-4].split('-')[2])

        # Predict result and print
        result = self.model.predict(image)[0][0]
        print("Predicted, Actual: %s, %s" % (result, label))

    
    def test(self, ds_dir="data/test"):
        test_data = self.fetch_dataset(ds_dir)
        results = self.model.evaluate(test_data)
        print("MSE (loss), MAE: ", results)


    def save(self, filename="model.h5"):
        self.model.save(filename)
        print("Model saved to ", filename)


    def plot_tsne(self, dataset, layer):
        # Produce a model that returns layer activations
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_outputs)

        images = []
        labels = []

        # Extract activations at specified layer for each image
        for batch in dataset.take(1).as_numpy_iterator():
            for i in range(len(batch[1])):
                image = tf.expand_dims(batch[0][i], 0)
                activations = activation_model.predict(image)
                images.append(tf.keras.backend.flatten(activations[layer]))
            labels.extend(batch[1])

        # Apply TSNE
        images_2d = TSNE(n_components=2, verbose=1).fit_transform(images)
        x, y = zip(*images_2d)

        # Set colour for each point based on BPM
        c = [(bpm - 50) / 150 for bpm in y]

        # Plot data
        plt.figure()
        plt.scatter(x, y, c=c)
        plt.show()


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) > 0:
        beatnet = BeatNet()
        if argv[0] == "train":
            beatnet.gen_model()
            beatnet.train(*argv[1:])
            beatnet.save()

        if argv[0] == "predict":
            beatnet.load_model()
            beatnet.predict(argv[1])

        if argv[0] == "test":
            beatnet.load_model()
            beatnet.test()

        if argv[0] == "tsne":
            beatnet.load_model()
            dataset = beatnet.fetch_dataset('data/training')
            beatnet.plot_tsne(dataset, 1)
