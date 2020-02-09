import tensorflow as tf
import os
import sys

def predict(image_path):
    model = tf.keras.models.load_model('model.h5')
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 64])
    image /= 255.0
    image = tf.reshape(image, (1, 256, 64, 3))

    result = model.predict(image)
    print(result[0][0] * 200)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 1:
        exit('Please specify an image file.')
    if (os.path.isfile(argv[0]) == False):
        exit('Image file not found: "%s"' % argv[0])
        
    predict(*argv)
