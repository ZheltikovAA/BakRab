import labirint
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

# Модификации GAN

# Определение генератора
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x


# Определение дискриминатора
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 2304)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# Проще переделать через

# Функция обучения GAN
def train_gan(data, num_epochs, batch_size, generator, discriminator, generator_optimizer, discriminator_optimizer,
              device):
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for i in range(0, len(data), batch_size):
            # real_inputs - изображения из набора данных MNIST
            # fake_inputs - изображения от генератора
            # real_inputs должны быть классифицированы как 1, а fake_inputs - как 0
            real_inputs = data[i:i + batch_size].to(device)
            real_outputs = discriminator(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_inputs = generator(noise)
            fake_outputs = discriminator(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)
            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)
            D_loss = criterion(outputs, targets)
            discriminator_optimizer.zero_grad()
            D_loss.backward()
            discriminator_optimizer.step()
            # Обучаем генератор
            # Цель генератора получить от дискриминатора 1 по всем изображениям
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_inputs = generator(noise)
            fake_outputs = discriminator(fake_inputs)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
            G_loss = criterion(fake_outputs, fake_targets)
            generator_optimizer.zero_grad()
            G_loss.backward()
            generator_optimizer.step()
            if i % 100 == 0 or i == len(data):
                print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, i,
                                                                                                      D_loss.item(),
                                                                                                      G_loss.item()))

    torch.save(generator.state_dict(), 'Generator_epoch.pth')


# Создаем экземпляры генератора и дискриминатора
input_size = 128
hidden_size = 256
output_size = 64 * 36
generator = Generator(input_size, hidden_size, output_size)
discriminator = Discriminator(output_size, hidden_size)


# Определяем устройство (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

# Определяем оптимизаторы
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Генерируем обучающие данные
training_data = []
for _ in range(256):
    training_data.append(labirint.labirint([[0] * 64 for _ in range(36)], 64, 36))
data = torch.tensor(np.array(training_data).astype('float32') >= 0.5).float()
# Обучаем GAN
num_epochs = 100
batch_size = 32
# train_gan(data, num_epochs, batch_size, generator, discriminator, generator_optimizer, discriminator_optimizer, device)

def get_maze_nn():
    generator.load_state_dict(torch.load('Generator_epoch.pth'))
    generator.eval()
    noise = (torch.rand(1, input_size) - 0.5) / 0.5
    maze = generator(noise)
    maze_np = maze.view(36, 64).detach().numpy()
    maze_np = np.rint(maze_np)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(np.sum(maze_np > 0.5))
    print(len(maze_np) , " ", len(maze_np[0]))
    return maze_np
