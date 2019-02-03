import os
import glob
import cv2
import torch
import random
import sys
import librosa
import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, data_dir, mode, length=3, fps=25, size=224, sr=16000):
        self.length = length
        self.size = size
        self.fps = fps
        self.sr = sr
        self.frame_offset = None
        if mode is 'train':
            subdir = 'train'
        elif mode is 'test':
            subdir = 'test'
        elif mode is 'val':
            subdir = 'val'
        elif mode is 'toy':
            subdir = 'toy'
        else:
            print('[!] Dataset mode error.')
            sys.exit()
        self.all_video = sorted(glob.glob(os.path.join(data_dir, subdir, 'original', 'cropped', '*.mp4')))
        self.all_audio = sorted(glob.glob(os.path.join(data_dir, subdir, 'numpy', 'audio', '*.npy')))

    def __getitem__(self, index):
        video_path = self.all_video[index]
        audio_path = self.all_audio[index]
        video = self.load_video(video_path)
        audio = self.load_audio(audio_path)
        return video, audio

    def __len__(self):
        return len(self.all_video)

    def load_video(self, video_path, margin=5):
        vc = cv2.VideoCapture(video_path)
        num_of_frame = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_of_frame < 75:
            print('[!] Frame length is under 75 at:', video_path)
            sys.exit()

        target_length = self.length * self.fps
        frames = torch.FloatTensor(target_length, 3, self.size, self.size)
        offset_boundary = (num_of_frame - margin) - target_length
        if offset_boundary < 0:
            offset_boundary = 0

        # [!][!][!][!] For toy test only [!][!][!][!]
        offset_boundary = 0

        self.frame_offset = random.randint(0, offset_boundary)

        for idx in range(self.frame_offset, self.frame_offset + target_length):
            frame = vc.read()[1]
            frame = torch.from_numpy(frame)
            # HWC2CHW
            frame = frame.permute(2, 0, 1)
            frames[idx - self.frame_offset, :, :, :] = frame
        frames /= 255
        return frames

    def load_audio(self, audio_path):
        samples = np.load(audio_path)
        offset_sec = self.frame_offset / self.fps
        sample_offset = int(self.sr * offset_sec)
        target_length = self.length * self.sr
        samples = samples[sample_offset:sample_offset + target_length]
        spect = self.get_spectrogram(samples)
        spect = np.transpose(spect, (0, 2, 1))
        return spect

    def get_spectrogram(self, y):
        D = librosa.stft(y, n_fft=512, hop_length=int(0.01*self.sr), win_length=int(0.025*self.sr), window='hann')
        r = D.real
        i = D.imag
        concat = np.stack((r, i), axis=0)
        return concat

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = Dataset('./data_toy', mode='train')
    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for step, (video, audio) in enumerate(loader):
        print('---')
        print(video.shape)  # (B, 75, 3, 224, 224)
        print(audio.shape)  # (B, 2, 301, 257)
        break
