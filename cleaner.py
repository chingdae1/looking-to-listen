import os
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./data_toy')
parser.add_argument('--result_out', type=str, default='./clean_list.txt')
args = parser.parse_args()

all_audio = glob.glob(os.path.join(args.data_dir, '*', 'numpy', 'audio', '*.npy'))
print(len(all_audio), 'audio is found.')
clean_list = []
for step, audio in enumerate(all_audio):
    if step % 1000 == 0:
        print(step, 'is done.')

    a = np.load(audio)
    length = a.shape[0]
    if length < 48000:
        basename = os.path.basename(audio)
        id = basename.replace('.npy', '\n')
        clean_list.append(id)

print('Write to file..')
with open(args.result_out, 'w') as f:
    for id in clean_list:
        f.write(id)
print(len(clean_list), 'audio have to be removed.')
print('done')