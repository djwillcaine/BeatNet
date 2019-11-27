import tensorflow as tf
import os
import sys

def predict(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 32])
    image /= 255.0
    image = tf.reshape(image, (1, 256, 32, 3))

    print(model.predict(image))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 2:
        exit('Program requires 2 parameters: path to model and path to image')
    if (os.path.isfile(argv[0]) == False):
        exit('Model file not found: "%s"' % argv[0])
    if (os.path.isfile(argv[1]) == False):
        exit('Image file not found: "%s"' % argv[1])
        
    predict(*argv)
