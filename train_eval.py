import os
import argparse
import pathlib
import subprocess

data_dir = 'data'
models_dir = 'models'
output_modes = ['classification', 'regression']
architectures = ['shallow', 'deep']

def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        # directory already exists
        pass

def train_eval(train_only, eval_only):

    if not eval_only:
        print('Training models...')
        for dataset in pathlib.Path(data_dir).iterdir():
            for o in output_modes:
                for a in architectures:
                    subprocess.run([
                        'python', 'train.py',
                        '-d', str(dataset),
                        '-a', a,
                        '-o', o
                    ])

    if not train_only:
        print('Evaluating models...')
        create_dir('eval')
        file = open(os.path.join('eval', 'results.csv'), 'w')
        file.write('Dataset,Model Architecture,Model Type,BPM Range,MSE,MAE,Accuracy1,Accuracy2\n')
        file.close()
            
        for model_path in pathlib.Path(models_dir).iterdir():
            if model_path.stem.split('.')[-1] == 'best':
                subprocess.run(['python', 'evaluate.py', str(model_path), '-w'])

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train-only', action='store_true')
    parser.add_argument('-e', '--evaluate-only', action='store_true')

    args = parser.parse_args()
