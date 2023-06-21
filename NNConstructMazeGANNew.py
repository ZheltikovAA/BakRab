import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import labirint
import numpy as np
import time

import plot_graph
from plot_graph import plt_graph, plt_gpu_use_grapf


def select_device(device):
    if device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA is not available")
    else:
        raise ValueError("Invalid device")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, input_dim=144):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_dim, 1440)
        self.bn1 = nn.BatchNorm1d(1440)
        self.relu1 = nn.ReLU()

        self.conv1 = nn.ConvTranspose2d(60, 1, kernel_size=(3, 4), stride=(3, 2), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu3 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(1, 1, kernel_size=(6, 6), stride=(2, 2), padding=1)
        self.sig = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = x.view(x.size(0), 60, 3, 8)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv4(x)
        x = self.sig(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(6, 6), stride=(2, 2), padding=1)
        self.leakyrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.leakyrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1)
        self.leakyrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(1, 60, kernel_size=(3, 4), stride=(3, 2), padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(60)
        self.leakyrelu4 = nn.LeakyReLU(0.2)

        self.fc = nn.Linear(60 * 3 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


def generate_train_data(n):
    start_time = time.time()
    training_data = []
    for _ in range(n):
        maze, loadavg, loadavg_all = labirint.labirint_ai([[0] * 64 for _ in range(36)], 64, 36)
        training_data.append(maze)
    data = torch.tensor(np.array(training_data).astype('float32') <= 0.5).float()
    for i in range(n):
        data[i][0][35][62] = 2.
    end_time = time.time()
    execution_time = end_time - start_time
    print("time data generate: ", execution_time)
    return data


def train(num_epochs, train_data, loss_fn, generator, discriminator, optimizer_generator, optimizer_discriminator,
          device, batch_size, scheduler_gen, scheduler_disc):
    generator.train()
    discriminator.train()
    d_loss_list = []
    g_loss_list = []
    memory_allocated = []
    memory_reserved = []
    for epoch in range(num_epochs):
        start_time = time.time()
        print('Memory Usage: Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB',
              'Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        for i in range(0, len(train_data), batch_size):
            discriminator.zero_grad()

            real_inputs = train_data[i:i + batch_size].to(device)
            real_outputs = discriminator(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)
            # Получаем шум и закидываем в генератор для получения ложных матриц
            # if epoch % 2 == 0:
            noise = (torch.randn(real_inputs.shape[0], 144)).to(device)

            fake_inputs = generator(noise)
            fake_outputs = discriminator(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)
            # Train the Discriminator
            d_loss = loss_fn(outputs, targets)

            d_loss.backward()
            optimizer_discriminator.step()
            generator.zero_grad()
            # Получаем шум и закидываем в генератор для получения ложных матриц
            noise = (torch.randn(real_inputs.shape[0], 144))
            noise = noise.to(device)  # Перемещение шума на выбранное устройство

            fake_inputs = generator(noise)
            fake_outputs = discriminator(fake_inputs)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)

            # Train the Generator
            g_loss = loss_fn(fake_outputs, fake_targets)

            g_loss.backward()
            optimizer_generator.step()
        memory_allocated.append(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
        memory_reserved.append(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

        # scheduler_disc.step()
        # scheduler_gen.step()
        time_spend = time.time() - start_time
        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item())
        print('Epoch [{}/{}], D_Loss: {:.3f}, G_Loss: {:.3f}, Time GPU: {:.3f}'
              .format(epoch + 1, num_epochs, d_loss.item(), g_loss.item(), time_spend))
        if epoch % 100 == 0:
            plt_gpu_use_grapf(memory_allocated,memory_reserved)
            plt_graph(d_loss_list, g_loss_list)
            print("save")
            torch.save(generator.state_dict(), 'D:\\BakRab\\DCGAN_Model_Tan_GPU_GRAPH_pow\\Generator_epoch_DCGAN_{}.pth'.format(epoch))
    plt_graph(d_loss_list, g_loss_list)
    print("save")
    torch.save(generator.state_dict(), 'D:\\BakRab\\DCGAN_Model_Tan_GPU_GRAPH_pow\\Generator_epoch_DCGAN_{}.pth'.format(num_epochs))


def get_data(input_dim_noize_vector):
    print(torch.cuda.get_device_name())
    start_time = time.time()
    device_new = torch.device('cuda:0')
    generator = Generator(144)
    generator.load_state_dict(torch.load('D:\\BakRab\\DCGAN_Model_Tan_GPU_GRAPH_pow\\Generator_epoch_DCGAN_450.pth', map_location=device_new))
    generator.to(device_new)
    generator.eval()
    noise = (torch.randn(1, 144)).to(device_new)
    noise = noise.to(device_new)
    maze = generator(noise)
    maze_np = maze.detach().cpu().numpy()
    maze_np = np.rint(maze_np)
    print(maze_np.shape)
    end_time = time.time()
    execution_time = end_time - start_time
    print("All time spent : {}".format(execution_time))
    return maze_np[0][0]


if __name__ == "__main__":
    num_epochs = 1000
    batch_size = 50000
    input_dim_noize_vector = 144
    n = 100000

    device = select_device("cuda")

    generator = Generator(144)
    discriminator = Discriminator()
    discriminator.apply(weights_init)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.002)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.002)

    scheduler_gen = StepLR(optimizer_gen, step_size=250, gamma=0.1)
    scheduler_disc = StepLR(optimizer_disc, step_size=250, gamma=0.1)
    loss_fn = nn.BCELoss()
    train_data = generate_train_data(n)
    print(train_data.shape)
    train(num_epochs, train_data, loss_fn, generator, discriminator, optimizer_gen, optimizer_disc,
          device, batch_size, scheduler_gen, scheduler_disc)
    #
    print(get_data(input_dim_noize_vector))
