import torch
from torch import nn
import labirint
import numpy as np
import time
import plot_graph
import numba


def generate_train_data(n):

    loadavg = []
    loadavg_all = []
    training_data = []
    for _ in numba.prange(n):
        maze, loadavg_, loadavg_all_ = labirint.labirint_ai([[0] * 64 for _ in numba.prange(36)], 64, 36)
        training_data.append(maze)
        loadavg.append(loadavg_)
        loadavg_all.append(loadavg_all_)
    data = torch.tensor(np.array(training_data).astype('float32') <= 0.5).float()
    for i in numba.prange(n):
        data[i][35][62] = 2.
    plot_graph.plt_cpuavg_use_graph(loadavg_all)
    return data


if __name__ == "__main__":
    n = 1000
    start = time.time()
    data = generate_train_data(n)
    end = time.time()
    execute = end - start
    print(execute, "<<--", n)

    n = 5000
    start = time.time()
    data = generate_train_data(n)
    end = time.time()
    execute = end - start
    print(execute, "<<--", n)

    n = 25000
    start = time.time()
    data = generate_train_data(n)
    end = time.time()
    execute = end - start
    print(execute, "<<--", n)

    n = 50000
    start = time.time()
    data = generate_train_data(n)
    end = time.time()
    execute = end - start
    print(execute, "<<--", n)

    n = 100000
    start = time.time()
    data = generate_train_data(n)
    end = time.time()
    execute = end - start
    print(execute, "<<--", n)


#     batch_size = 10
#     input_dim = 144
#
#
#
#     data = generate_train_data(batch_size)
#     noise = (torch.randn(data.shape[0], 144))
#     #
#     print(noise.shape)
#
#     fc1 = nn.Linear(input_dim, 1440)
#     noise = fc1(noise)
#     print(noise.shape, " fc1")
#
#     bn1 = nn.BatchNorm1d(1440)
#     noise = bn1(noise)
#     print(noise.shape, " bn1")
#
#     relu1 = nn.ReLU()
#     noise = relu1(noise)
#     print(noise.shape, " relu1")
#     noise = noise.view(noise.size(0), 60, 3, 8)
#     print(noise.shape, " view")
#
#     conv1 = nn.ConvTranspose2d(60, 1, kernel_size=(3, 4), stride=(3, 2), padding=(0, 1))
#     noise = conv1(noise)
#     print(noise.shape, " conv1")
#
#     bn2 = nn.BatchNorm2d(1)
#     noise = bn2(noise)
#     print(noise.shape, " bn2")
#
#     relu2 = nn.ReLU()
#     noise = relu2(noise)
#     print(noise.shape, " relu2")
#
#     conv2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=1)
#     bn3 = nn.BatchNorm2d(1)
#     relu3 = nn.ReLU()
#
#     noise = conv2(noise)
#     print(noise.shape, " conv2")
#     noise = bn3(noise)
#     print(noise.shape, " bn3")
#     noise = relu3(noise)
#     print(noise.shape, " relu3")
#
#     conv4 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=1)
#     bn5 = nn.BatchNorm2d(1)
#     relu5 = nn.ReLU()
#
#     noise = conv4(noise)
#     print(noise.shape, " conv4")
#     noise = bn5(noise)
#     print(noise.shape, " bn5")
#     noise = relu5(noise)
#     print(noise.shape, " relu5")
#
#     conv3 = nn.ConvTranspose2d(1, 1, kernel_size=(6, 6), stride=(2, 2), padding=1)
#     noise = conv3(noise)
#     print(noise.shape, " conv3")
#
#     sig = nn.Sigmoid()
#     noise = sig(noise)
#     print(noise.shape, " tanh")
#     threshold = 0.5
#     binary_output = torch.where(noise > threshold, torch.ones_like(noise), torch.zeros_like(noise))
#
#     print(binary_output)
#     print(binary_output.shape)
#
# #     print("-------------------------------\nDiscriminator")
# #     noise = data
# #     # ----------------------------------------------------------------
# #     # Discriminator
# #     print(noise.shape)
# #     conv1 = nn.Conv2d(1, 1, kernel_size=(6, 6), stride=(2, 2), padding=1)
# #     noise = conv1(noise)
# #     print(noise.shape, " conv1")
# #
# #     leakyrelu1 = nn.LeakyReLU(0.2)
# #     noise = leakyrelu1(noise)
# #     print(noise.shape, " leakyrelu1")
# #
# #     conv2 = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=1)
# #     noise = conv2(noise)
# #     print(noise.shape, " conv2")
# #
# #     bn2 = nn.BatchNorm2d(1)
# #     noise = bn2(noise)
# #     print(noise.shape, " bn2")
# #
# #     leakyrelu2 = nn.LeakyReLU(0.2)
# #     noise = leakyrelu2(noise)
# #     print(noise.shape, " leakyrelu2")
# #
# #     conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=1)
# #     noise = conv3(noise)
# #     print(noise.shape, " conv3")
# #
# #     bn3 = nn.BatchNorm2d(1)
# #     noise = bn3(noise)
# #     print(noise.shape, " bn3")
# #
# #     leakyrelu3 = nn.LeakyReLU(0.2)
# #     noise = leakyrelu3(noise)
# #     print(noise.shape, " leakyrelu3")
# #
# #     conv4 = nn.Conv2d(1, 60, kernel_size=(3, 4), stride=(3, 2), padding=(0, 1))
# #     noise = conv4(noise)
# #     print(noise.shape, " conv4")
# #
# #     bn4 = nn.BatchNorm2d(60)
# #     noise = bn4(noise)
# #     print(noise.shape, " bn4")
# #
# #     leakyrelu4 = nn.LeakyReLU(0.2)
# #     noise = leakyrelu4(noise)
# #     print(noise.shape, " leakyrelu4")
# #
# #     noise = noise.view(noise.size(0), -1)
# #     print(noise.shape, "view")
# #     fc = nn.Linear(60*3*8, 1)
# #     noise = fc(noise)
# #     print(noise.shape, " fc")
# #
# #     sigmoid = nn.Sigmoid()
# #     noise = sigmoid(noise)
# #     print(noise.shape)
# #     print(noise)
# #
# #
# #     print(torch.ones(batch_size, 1))
# # #