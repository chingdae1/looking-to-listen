import torch
import torch.nn as nn
import model
from dataset import Dataset
import os
from torch.utils.data import DataLoader
from submodule import vgg_face_dag


class Solver():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = Dataset(data_dir=config['data_dir'],
                                  mode='train')
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       shuffle=True,
                                       drop_last=True)
        self.test_data = Dataset(data_dir=config['data_dir'],
                                 mode='test')
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=config['batch_size'],
                                      num_workers=config['num_workers'],
                                      shuffle=True,
                                      drop_last=True)
        self.val_data = Dataset(data_dir=config['data_dir'],
                                mode='val')
        self.val_loader = DataLoader(self.val_data,
                                     batch_size=config['batch_size'],
                                     num_workers=config['num_workers'],
                                     shuffle=True,
                                     drop_last=True)
        self.net = model.Net(config['num_of_face'], self.device).to(self.device)
        if config['load_model']:
            print('Load pretrained model..')
            state_dict = torch.load(config['load_path'])
            self.net.load_state_dict(state_dict)
        if config['multi_gpu']:
            print('Use Multi GPU')
            self.net = torch.nn.DataParallel(self.net, device_ids=config['gpu_ids'])
        self.MSE = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(),
                                      lr=config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim,
                                                                    factor=config['factor'],
                                                                    patience=config['patience'],
                                                                    verbose=True)
        self.vgg_face = vgg_face_dag(config['vgg_face_path'])
        self.vgg_face.eval()
        self.optim_vgg = torch.optim.Adam(self.vgg_face.parameters(),
                                          lr=config['lr'])
        self.scheduler_vgg = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim_vgg,
                                                                        factor=config['factor'],
                                                                        patience=config['patience'],
                                                                        verbose=True)
        for param in self.vgg_face.parameters():
            param.requires_grad = False
        num_ftrs = self.vgg_face.fc8.in_features
        self.vgg_face.fc8 = nn.Linear(num_ftrs, 1024)
        if config['load_vgg']:
            print('Load pretrained vgg face..')
            state_dict = torch.load(config['load_vgg_path'])
            self.vgg_face.load_state_dict(state_dict)
        self.vgg_face = self.vgg_face.to(self.device)
        self.saved_dir = os.path.join(config['save_dir'], config['model_name'])
        os.makedirs(self.saved_dir, exist_ok=True)

    def fit(self):
        print('Start training..')
        for epoch in range(self.config['epoch']):
            video_list = []
            audio_list = []
            face_embedding_list = []
            for step, (video, audio, _) in enumerate(self.train_loader):
                if (step + 1) % self.config['num_of_face'] != 0:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                else:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                    audio_mix = 0
                    for idx in range(self.config['num_of_face']):
                        one_face_list = []
                        for video in video_list[idx]:
                            one_face_embedding = self.vgg_face(video)
                            one_face_list.append(one_face_embedding)
                        face_embedding = torch.stack(one_face_list, dim=0)
                        face_embedding = face_embedding.view(-1, 1024, 75, 1)
                        face_embedding_list.append(face_embedding)
                        audio_mix += audio_list[idx]
                    # audio_mix = Dataset.power_law_compression(audio_mix)
                    masks = self.net(face_embedding_list, audio_mix)
                    print(masks.shape)
                    separated_list = []
                    for mask in masks:
                        print(audio_mix.shape)
                        print(mask.shape)
                        separated = audio_mix * mask
                        separated_list.append(separated)
                    final_output = torch.stack(separated_list, dim=1)
                    ground_truth = torch.stack(audio_list, dim=1)  # (N, F, 2, 301, 257)
                    # ground_truth = Dataset.power_law_compression(ground_truth)
                    loss = self.MSE(final_output, ground_truth)
                    self.optim.zero_grad()
                    self.optim_vgg.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.optim_vgg.step()

                    print('Epoch[{}/{}]  Step[{}/{}]  Loss: {:.8f}'.format(
                        epoch + 1, self.config['epoch'], step + 1,
                        self.train_data.__len__() // self.config['batch_size'],
                        loss.item()
                    ))

                    video_list = []
                    audio_list = []
                    face_embedding_list = []
            if (epoch + 1) % self.config['val_every'] == 0:
                val_loss = self.validation(epoch + 1)
                self.scheduler.step(val_loss)
                self.scheduler_vgg.step(val_loss)
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save(epoch)

    def validation(self, epoch):
        print('Start validation..')
        self.net.eval()
        video_list = []
        audio_list = []
        face_embedding_list = []
        idx_list = []
        total_loss = 0
        cnt = 0
        with torch.no_grad():
            for step, (video, audio, index) in enumerate(self.val_loader):
                if (step + 1) % self.config['num_of_face'] != 0:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                    idx_list.append(index)
                else:
                    video_list.append(video.to(self.device))
                    audio_list.append(audio.to(self.device))
                    idx_list.append(index)
                    audio_mix = 0
                    for idx in range(self.config['num_of_face']):
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
                    total_loss += loss
                    print('[val] Step[{}/{}]  Loss: {:.8f}'.format(
                        step + 1,
                        self.val_data.__len__() // self.config['batch_size'],
                        loss.item()
                    ))

                    if step < self.config['sample_for']:
                        idx_tensor = torch.stack(idx_list, dim=1)
                        vid_tensor = torch.stack(video_list, dim=1)  # (N, F, 75, 3, 224, 224)
                        self.get_sample(step + 1, epoch, audio_mix, final_output, ground_truth, vid_tensor, idx_tensor)

                    video_list = []
                    audio_list = []
                    face_embedding_list = []
                    idx_list = []
                    cnt += 1

            average_loss = total_loss / cnt
            print('[Validation {}] Average Loss: {:.8f}'.format(epoch, average_loss))
        self.net.train()
        return average_loss

    def get_sample(self, step, epoch, audio_mix, final_output, ground_truth, video, idx_tensor):
        sample_dir = os.path.join(self.config['val_sample_dir'], self.config['model_name'], 'epoch_' + str(epoch), 'step_' + str(step))
        os.makedirs(os.path.join(sample_dir), exist_ok=True)

        for i, separated in enumerate(final_output):
            batch_dir = os.path.join(sample_dir, 'batch_' + str(i))
            os.makedirs(batch_dir, exist_ok=True)
            mix = audio_mix[i]  # (2, 301, 257)
            mix_path = os.path.join(batch_dir, 'mix.wav')
            Dataset.spect_to_wav(mix, mix_path)
            gt = ground_truth[i]
            vid_tensor = video[i]
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
                gt_path = os.path.join(gt_dir, data_id + '.wav')
                output_path = os.path.join(output_dir, data_id + '.wav')
                video_path = os.path.join(video_dir, data_id + '.mp4')

                # gt[k] = Dataset.decompression(gt[k])
                # s = Dataset.decompression(s.detach().cpu().numpy())
                Dataset.spect_to_wav(gt[k], gt_path)
                Dataset.spect_to_wav(s.detach().cpu().numpy(), output_path)
                Dataset.tensor_to_vid(vid_tensor[k], video_path)

    def save(self, epoch):
        checkpoint = {
            'net': self.net.state_dict()
        }
        checkpoint_vgg = {
            'net': self.vgg_face.state_dict()
        }
        output_path = os.path.join(self.saved_dir, 'model_' + str(epoch) + '.pt')
        output_path_vgg = os.path.join(self.saved_dir, 'model_vgg_' + str(epoch) + '.pt')
        torch.save(checkpoint, output_path)
        torch.save(checkpoint_vgg, output_path_vgg)
