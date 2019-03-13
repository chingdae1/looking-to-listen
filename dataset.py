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

        ## TOY
        # self.all_video = self.all_video[:100]
        # self.all_audio = self.all_audio[:100]

        print(len(self.all_video), 'video has been found.')
        print(len(self.all_audio), 'audio has been found.')

        if len(self.all_video) != len(self.all_audio):
            print('[!] The number of video/audio is not same.')
            sys.exit()

    def __getitem__(self, index):
        video_path = self.all_video[index]
        audio_path = self.all_audio[index]
        try:
            video = self.load_video(video_path)
            audio = self.load_audio(audio_path)
            return video, audio, index
        except:
            print('[!] DATA ERROR')
            print(video_path)
            print(audio_path)
            print('FRAME_OFFSET:', self.frame_offset)
            sys.exit()

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
        # offset_boundary = 0

        self.frame_offset = random.randint(0, offset_boundary)
        vc.set(cv2.CAP_PROP_POS_FRAMES, self.frame_offset)
        for idx in range(0, target_length):
            frame = vc.read()[1]
            frame = torch.from_numpy(frame)
            # HWC2CHW
            frame = frame.permute(2, 0, 1)
            frame[0, :, :] -= 129.186279296875
            frame[1, :, :] -= 104.76238250732422
            frame[2, :, :] -= 93.59396362304688
            frames[idx, :, :, :] = frame
        # frames /= 255
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

    def get_id_by_idx(self, idx_tensor):
        idx = idx_tensor.item()
        data_id = os.path.basename(self.all_video[idx]).replace('.mp4', '')
        return data_id

    @staticmethod
    def spect_to_wav(spect, output_path, sr=16000, hop_length=160, win_length=400):
        complex = np.ndarray((spect.shape[1], spect.shape[2]), dtype=np.complex)
        complex.real = spect[0]
        complex.imag = spect[1]
        complex = np.transpose(complex, (1, 0))
        y_hat = librosa.istft(complex, hop_length=hop_length, win_length=win_length, length=48000)
        librosa.output.write_wav(output_path, y_hat, sr=sr)

    @staticmethod
    def tensor_to_vid(vid_tensor, output_path, fps=25.0, fourcc='mp4v', size=(224, 224)):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))

        for frame in vid_tensor:
            frame = frame.permute(1, 2, 0).cpu().numpy()
            # frame *= 255
            frame[:, :, 0] += 129.186279296875
            frame[:, :, 1] += 104.76238250732422
            frame[:, :, 2] += 93.59396362304688
            frame = frame.astype(np.uint8)
            vid_writer.write(frame)

    @staticmethod
    def power_law_compression(spect, p=0.3):
        return spect ** p

    @staticmethod
    def decompression(spect, p=0.3):
        return spect ** (1 / p)

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = Dataset('./data_toy', mode='toy')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

    index = 0
    for step, (video, audio, index) in enumerate(loader):
        print(video.shape)  # (N, 75, 3, 224, 224)
        print(audio.shape)  # (N, 2, 301, 257)
        break
