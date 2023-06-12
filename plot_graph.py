import matplotlib.pyplot as plt
import numpy as np

def plt_graph(d_loss, g_loss):
    fig, ax = plt.subplots()

    # Построение графика для g_loss
    ax.plot(range(len(g_loss)), g_loss, label='Generator Loss')

    # Построение графика для d_loss
    ax.plot(range(len(d_loss)), d_loss, label='Discriminator Loss')

    # Настройка заголовка и меток осей
    ax.set_title('Generator and Discriminator Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Добавление легенды
    ax.legend()
    plt.grid()
    # Отображение графика
    plt.show()
