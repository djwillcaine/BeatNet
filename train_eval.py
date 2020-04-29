import os
import pathlib
import subprocess

data_dir = 'data'
models_dir = 'models'
output_modes = ['classification', 'regression']
architectures = ['shallow', 'deep']


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

print('Evaluating models...')
for model_path in pathlib.Path(models_dir).iterdir():
    if model_path.stem.split('.')[-1] == 'best':
        subprocess.run(['python', 'evaluate.py', str(model_path), '-w'])

print('Done.')