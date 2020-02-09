import tensorflow as tf
import os
import sys
import pathlib

def evaluate(eval_set_path='eval_set', model_path='model.h5'):
    if (os.path.isdir(eval_set_path) == False):
        exit('Image file not found: "%s"' % argv[0])

    model = tf.keras.models.load_model(model_path)
    
    data_root = pathlib.Path(eval_set_path)
    files = data_root.iterdir()
    
    results = []
    p99th = 0
    p95th = 0
    p70th = 0
    
    for filename in files:
        filename = str(filename)
        
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [256, 64])
        image /= 255.0
        image = tf.reshape(image, (1, 256, 64, 3))

        predicted = round(model.predict(image)[0][0], 2)
        bpm = float(filename.split('-')[2][:-4])
        
        if predicted >= bpm * 0.7 and predicted <= bpm * 1.3:
            p70th = p70th + 1
        if predicted >= bpm * 0.95 and predicted <= bpm * 1.05:
            p95th = p95th + 1
        if predicted >= bpm * 0.99 and predicted <= bpm * 1.01:
            p99th = p99th + 1

        print ('Predicted: %f\t| Actual: %f' % (predicted, bpm))

    print('99th percentile: %f' % p99th)
    print('95th percentile: %f' % p95th)
    print('70th percentile: %f' % p70th)
        

if __name__ == "__main__":
    argv = sys.argv[1:]
    evaluate(*argv)
