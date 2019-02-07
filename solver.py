import torch
import torch.nn as nn
import model
from dataset import Dataset
import os
import librosa
import numpy as np
from torch.utils.data import DataLoader
from submodule import vgg_face_dag
from shutil import copyfile


class Solver():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = Dataset(data_dir=config.data_dir,
                                  mode='toy')
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=True,
                                       drop_last=True)
        self.test_data = Dataset(data_dir=config.data_dir,
                                 mode='test')
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=True,
                                      drop_last=True)
        self.val_data = Dataset(data_dir=config.data_dir,
                                mode='toy')
        self.val_loader = DataLoader(self.val_data,
                                     batch_size=config.batch_size,
                                     num_workers=config.num_workers,
                                     shuffle=True,
                                     drop_last=True)
        self.net = model.Net(config.num_of_face, self.device).to(self.device)
        self.MSE = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(),
                                      lr=config.lr)
        self.vgg_face = vgg_face_dag(config.vgg_face_path)
        self.vgg_face.eval()
        for param in self.vgg_face.parameters():
            param.requires_grad = False
        num_ftrs = self.vgg_face.fc8.in_features
        self.vgg_face.fc8 = nn.Linear(num_ftrs, 1024)
        self.vgg_face = self.vgg_face.to(self.device)

    def fit(self):
        print('Start training..')
        for epoch in range(self.config.epoch):
            video_list = []
            audio_list = []
            face_embedding_list = []
            for step, (video, audio, _) in enumerate(self.train_loader):
                if (step + 1) % self.config.num_of_face != 0:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                else:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                    audio_mix = 0
                    for idx in range(self.config.num_of_face):
                        one_face_list = []
                        for video in video_list[idx]:
                            one_face_embedding = self.vgg_face(video)
                            one_face_list.append(one_face_embedding)
                        face_embedding = torch.stack(one_face_list, dim=0)
                        face_embedding = face_embedding.view(-1, 1024, 75, 1)
                        face_embedding_list.append(face_embedding)
                        audio_mix += audio_list[idx]
                    masks = self.net(face_embedding_list, audio_mix)
                    separated_list = []
                    for mask in masks:
                        separated = audio_mix * mask
                        separated_list.append(separated)
                    final_output = torch.stack(separated_list, dim=1)
                    ground_truth = torch.stack(audio_list, dim=1)  # (N, F, 2, 301, 257)
                    loss = self.MSE(final_output, ground_truth)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    print('Epoch[{}/{}]  Step[{}/{}]  Loss: {:.8f}'.format(
                        epoch + 1, self.config.epoch, step + 1,
                        self.train_data.__len__() // self.config.batch_size,
                        loss.item()
                    ))

                    video_list = []
                    audio_list = []
                    face_embedding_list = []
            if (epoch + 1) % self.config.val_every == 0:
                self.validation(epoch + 1)

    def validation(self, epoch):
        print('Start validation..')
        self.net.eval()
        video_list = []
        audio_list = []
        face_embedding_list = []
        idx_list = []
        loss_list = []
        for step, (video, audio, idx) in enumerate(self.val_loader):
            if (step + 1) % self.config.num_of_face != 0:
                video_list.append(video.to(self.device))
                audio_list.append(audio.to(self.device))
                idx_list.append(idx)
            else:
                video_list.append(video.to(self.device))
                audio_list.append(audio.to(self.device))
                idx_list.append(idx)
                audio_mix = 0
                for idx in range(self.config.num_of_face):
                    one_face_list = []
                    for video in video_list[idx]:
                        one_face_embedding = self.vgg_face(video)
                        one_face_list.append(one_face_embedding)
                    face_embedding = torch.stack(one_face_list, dim=0)
                    face_embedding = face_embedding.view(-1, 1024, 75, 1)
                    face_embedding_list.append(face_embedding)
                    audio_mix += audio_list[idx]
                masks = self.net(face_embedding_list, audio_mix)
                separated_list = []
                for mask in masks:
                    separated = audio_mix * mask
                    separated_list.append(separated)
                final_output = torch.stack(separated_list, dim=1)
                ground_truth = torch.stack(audio_list, dim=1)  # (N, F, 2, 301, 257)
                loss = self.MSE(final_output, ground_truth)
                loss_list.append(loss)
                print('Step[{}/{}]  Loss: {:.8f}'.format(
                    step + 1,
                    self.val_data.__len__() // self.config.batch_size,
                    loss.item()
                ))

                if step < self.config.sample_for:
                    idx_tensor = torch.stack(idx_list, dim=1)
                    self.get_sample(epoch, step + 1, audio_mix, final_output, idx_tensor)

                video_list = []
                audio_list = []
                face_embedding_list = []
                idx_list = []
        # FIX it !!!!!!
        # average_loss = np.average(loss_list)
        # print('[Validation {}] Average Loss: {:.8f}'.format(epoch, average_loss))
            self.net.train()

    def get_sample(self, step, epoch, audio_mix, final_output, idx_tensor):
        sample_dir = os.path.join(self.config.val_sample_dir, str(epoch) + '_epoch', 'step_' + str(step))
        os.makedirs(os.path.join(sample_dir), exist_ok=True)

        for i, separated in enumerate(final_output):
            batch_dir = os.path.join(sample_dir, 'batch_' + str(i))
            os.makedirs(batch_dir, exist_ok=True)
            mix = audio_mix[i]  # (2, 301, 257)
            mix_path = os.path.join(batch_dir, 'mix.wav')
            Solver.spect_to_wav(mix, mix_path)
            idx_batch = idx_tensor[i]

            for k, s in enumerate(separated):
                gt_dir = os.path.join(batch_dir, 'groud_truth')
                output_dir = os.path.join(batch_dir, 'output')
                video_dir = os.path.join(batch_dir, 'video')
                os.makedirs(gt_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(video_dir, exist_ok=True)

                idx = idx_batch[k]
                data_id = self.val_data.get_id_by_idx(idx)
                gt_path = self.val_data.get_aud_path_by_idx(idx)
                gt_sample_path = os.path.join(gt_dir, data_id + '.wav')
                output_path = os.path.join(output_dir, data_id + '.wav')
                video_path = self.val_data.get_vid_path_by_idx(idx)
                video_sample_path = os.path.join(video_dir, data_id + '.wav')

                copyfile(gt_path, gt_sample_path)
                Solver.spect_to_wav(s.detach().cpu().numpy(), output_path)
                copyfile(video_path, video_sample_path)

    @staticmethod
    def spect_to_wav(spect, output_path, sr=16000, hop_length=160, win_length=400):
        complex = np.ndarray((spect.shape[1], spect.shape[2]), dtype=np.complex)
        complex.real = spect[0]
        complex.imag = spect[1]
        complex = np.transpose(complex, (1, 0))
        y_hat = librosa.istft(complex, hop_length=hop_length, win_length=win_length, length=48000)
        librosa.output.write_wav(output_path, y_hat, sr=sr)
