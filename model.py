import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=96, kernel_size=(1, 7), padding=(0, 3), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(7, 1), padding=(3, 0), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(4, 2), dilation=(2, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(8, 2), dilation=(4, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(16, 2), dilation=(8, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(32, 2), dilation=(16, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(64, 2), dilation=(32, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(4, 4), dilation=(2, 2)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(8, 8), dilation=(4, 4)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(16, 16), dilation=(8, 8)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(32, 32), dilation=(16, 16)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), padding=(64, 64), dilation=(32, 32)),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=8, kernel_size=(1, 1), padding=(0, 0), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU()
        )

    def forward(self, x):
        return self.body(x)  # (N, 8, 301, 257)


class VideoStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(7, 1), padding=(3, 0), dilation=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(2, 0), dilation=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(4, 0), dilation=(2, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(8, 0), dilation=(4, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(16, 0), dilation=(8, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), padding=(32, 0), dilation=(16, 1)),
            nn.BatchNorm2d(256), nn.ReLU()
        )

    def nn_interpolation(self, x, target_dim=301):
        # def get_neighbor_list(target_dim):
        #     neighbor_list = [0 for _ in range(target_dim)]
        #     x_shape = x.shape
        #     for row in range(target_dim):
        #         neighbor_list[row] = int(row * x_shape[2] / target_dim)
        #     return neighbor_list

        # neighbor_list is hard coded, because it's always same.
        neighbor_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9,
                         10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19,
                         20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29,
                         30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39,
                         40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49,
                         50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59,
                         60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 66, 67, 67, 67, 67, 68, 68, 68, 68, 69, 69, 69, 69,
                         70, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 74, 74, 74, 74]
        x_shape = x.shape
        y = torch.empty((x_shape[0], x_shape[1], target_dim, x_shape[3]))
        for i, neighbor in enumerate(neighbor_list):
            y[:, :, i, :] = x[:, :, neighbor, :]
        return y

    def forward(self, x):
        x = self.body(x)  # (N, 256, 75, 1)
        x = self.nn_interpolation(x)  # (N, 256, 301, 1)
        return x


class Net(nn.Module):
    def __init__(self, num_of_face, device):
        super().__init__()
        self.video_stream = VideoStream()
        self.audio_stream = AudioStream()
        self.num_of_face = num_of_face
        self.device = device
        input_dim = (8 * 257) + (256 * num_of_face)
        self.BLSTM = nn.LSTM(input_size=input_dim, hidden_size=200, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(in_features=400, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=600)
        self.fc3 = nn.Linear(in_features=600, out_features=2*257*num_of_face)

    def forward(self, face_embedding_list, spectrogram):
        video_stream_output_list = []
        for face_embedding in face_embedding_list:
            video_stream_output = self.video_stream(face_embedding).to(self.device)
            video_stream_output_list.append(video_stream_output)
        audio_stream_output = self.audio_stream(spectrogram)
        audio_stream_output = audio_stream_output.permute(0, 1, 3, 2)  # (N, 8, 257, 301)
        audio_stream_output = audio_stream_output.contiguous().view((-1,
                                                        audio_stream_output.shape[1] * audio_stream_output.shape[2],
                                                        audio_stream_output.shape[3]))  # (N, 8*257, 301)
        video_stream_cat = torch.cat(video_stream_output_list, dim=1)  # (N, 256*F, 301, 1)
        video_stream_cat = video_stream_cat.view((-1, video_stream_cat.shape[1], video_stream_cat.shape[2]))
        av_fusion = torch.cat([video_stream_cat, audio_stream_output], dim=1)
        av_fusion = av_fusion.permute(0, 2, 1)  # (N, 301, 8*257 + 256*F)
        lstm_output, _ = self.BLSTM(av_fusion)  # (N, 301, 400)
        x = F.relu(lstm_output)
        x_out_list = []
        for i in range(x.shape[1]):
            x_out = self.fc1(x[:, i, :])
            x_out = F.relu(x_out)
            x_out = self.fc2(x_out)
            x_out = F.relu(x_out)
            x_out = self.fc3(x_out)
            x_out = torch.sigmoid(x_out)
            x_out_list.append(x_out)
        x = torch.stack(x_out_list, dim=1)  # (N, 301, F*514)
        spec_size = 2*257
        mask_list = []
        for i in range(self.num_of_face):
            start = i * spec_size
            mask = x[:, :, start:start+spec_size]
            mask_list.append(mask)
        x = torch.stack(mask_list, dim=0)
        x = x.permute(1, 0, 2, 3)  # (N, F, 301, 514)
        x = x.view(-1, self.num_of_face, 301, 2, 257)
        x = x.permute(0, 1, 3, 2, 4)  # (N, F, 2, 301, 257)
        return x


if __name__ == '__main__':
    face_embedding_1 = torch.empty((1, 1024, 75, 1))
    # face_embedding_2 = torch.empty((1, 1024, 75, 1))
    # face_embedding_3 = torch.empty((1, 1024, 75, 1))
    # face_embedding_4 = torch.empty((1, 1024, 75, 1))
    # face_embedding_list = [face_embedding_1, face_embedding_2, face_embedding_3, face_embedding_4]
    face_embedding_list = [face_embedding_1, face_embedding_1]
    audio_embedding = torch.empty((1, 2, 301, 257))
    device = torch.device('cpu')
    net = Net(len(face_embedding_list), device)
    net(face_embedding_list, audio_embedding)
