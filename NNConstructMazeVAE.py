import labirint
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.autograd import Variable

# Задаем размерность латентного пространства
latent_dim = 32


# Определяем класс энкодера
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
        x = nn.functional.sigmoid(self.deconv2(x))
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


# Создаем экземпляр модели VAE
vae = VAE()


# Определяем функцию потерь
def vae_loss(x_recon, x, z_mean, z_log_var):
    reconstruction_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + kl_loss


def train_model(num_epochs, num_samples):
    for epoch in range(num_epochs):
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i + batch_size]
            optimizer.zero_grad()
            x_recon, z_mean, z_log_var = vae(batch_data)
            loss = vae_loss(x_recon, batch_data, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    torch.save(vae.state_dict(), 'trained_model.pth')


def get_data():
    vae.load_state_dict(torch.load('trained_model.pth'))
    # Пример генерации новых данных
    sample_z = torch.randn(1, latent_dim)
    generated_data = vae.decoder(sample_z)
    maze = generated_data.detach().numpy()
    maze = np.rint(maze)
    print(maze.shape)
    print(maze[0][0])
    return maze[0][0]


# Задаем оптимизатор и функцию потерь
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Генерируем случайные данные для обучения (аналогичные матрицы)
num_samples = 256
training_data = []
for _ in range(num_samples):
    training_data.append(labirint.labirint([[0] * 64 for _ in range(36)], 64, 36))
data = torch.tensor(np.array(training_data).astype('float32') >= 0.5).float()

# data = torch.randn(num_samples, 1, 36, 64)
print(data.shape)
# Обучаем модель
num_epochs = 200
batch_size = 32
train_model(num_epochs, num_samples)

get_data()
