import torch
import torch.nn as nn
import model
from dataset import Dataset
from torch.utils.data import DataLoader
from submodule import vgg_face_dag


class Solver():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = Dataset(data_dir=config.data_dir,
                                  mode='toy ')
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
            for step, (video, audio) in enumerate(self.train_loader):
                if (step + 1) % self.config.num_of_face != 0:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                else:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                    audio_mix = 0
                    for idx in range(self.config.num_of_face):
                        one_face_list = []
                        print('vgg_face')
                        for video in video_list[idx]:
                            one_face_embedding = self.vgg_face(video)
                            one_face_list.append(one_face_embedding)
                        face_embedding = torch.stack(one_face_list, dim=0)
                        face_embedding = face_embedding.view(-1, 1024, 75, 1)
                        face_embedding_list.append(face_embedding)
                        audio_mix += audio_list[idx]
                    print('net')
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
                        self.train_data.__len__() // self.config.batch_size
                        , loss.item()
                    ))

                    video_list = []
                    audio_list = []
                    face_embedding_list = []
