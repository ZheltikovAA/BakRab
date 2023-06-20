import labirint
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import psutil as ps
import time
import plot_graph

latent_dim = 128


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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(9 * 16 * 64, latent_dim)
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var


# Определяем класс декодера
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 9 * 16 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = nn.functional.relu(self.fc1(z))
        x = x.view(x.size(0), 64, 9, 16)
        x = nn.functional.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x


# Определяем класс VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var


vae = VAE()


# Определяем функцию потерь
def vae_loss(x_recon, x, z_mean, z_log_var):
    reconstruction_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + kl_loss


def train_model(num_epochs, num_samples, device):
    device = select_device(device)
    vae.to(device)
    for epoch in range(num_epochs):
        start_time = time.time()
        loadavg = ps.cpu_percent(interval=None, percpu=False)
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i + batch_size].to(device)
            optimizer.zero_grad()
            x_recon, z_mean, z_log_var = vae(batch_data)
            loss = vae_loss(x_recon, batch_data, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
        ex_time = time.time() - start_time
        loadavg = ps.cpu_percent(interval=None, percpu=False) - loadavg
        print('Epoch [{}/{}], Loss: {:.3f}, Time: {:.3f}, Load CPU: {:.3f}'
              .format(epoch + 1, num_epochs, loss.item(), ex_time, loadavg))
    torch.save(vae.state_dict(), 'trained_model.pth')


def get_data(device):
    device = select_device(device)
    vae = VAE()
    vae.load_state_dict(torch.load('trained_model.pth', map_location=device))
    vae.to(device)
    # Пример генерации новых данных
    sample_z = torch.randn(1, latent_dim).to(device)
    generated_data = vae.decoder(sample_z)
    maze = generated_data.detach().cpu().numpy()
    maze[0][0][35][62] = 2
    maze = np.rint(maze)
    return maze[0][0]


if __name__ == "__main__":
    # Задаем оптимизатор и функцию потерь
    optimizer = optim.Adam(vae.parameters(), lr=0.0003)

    # Генерируем случайные данные для обучения (аналогичные матрицы)
    num_samples = 1000
    training_data = []
    loadavg = []
    loadavg_all = []
    start = time.time()
    for _ in range(num_samples):
        maze, loadavg_cpu, loadavg_ = labirint.labirint([[0] * 64 for _ in range(36)], 64, 36)
        training_data.append(maze)
        loadavg.append(loadavg_cpu)
        loadavg_all.append(loadavg_)
    data = torch.tensor(np.array(training_data).astype('float32') >= 0.5).float()
    for i in range(num_samples):
        data[i][0][35][62] = 2.
    print("spend_time: ", time.time() - start)
    print(data.shape)
    plot_graph.plt_cpu_use_graph(loadavg)
    # Обучаем модель
    num_epochs = 2000
    batch_size = 250
    print(sum(loadavg_all) / num_samples)
    train_model(num_epochs, num_samples, "cuda")
