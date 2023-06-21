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


def plt_graph_vae(loss):
    fig, ax = plt.subplots()

    # Построение графика для g_loss
    ax.plot(range(len(loss)), loss, label='Loss')

    # Настройка заголовка и меток осей
    ax.set_title('VAE Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Добавление легенды
    ax.legend()
    plt.grid()
    # Отображение графика
    plt.show()


def plt_gpu_use_grapf(allocated, reserved):
    fig, ax = plt.subplots()

    # Построение графика для g_loss
    ax.plot(range(len(allocated)), allocated, label='cuda.memory_allocated')

    # Построение графика для d_loss
    ax.plot(range(len(reserved)), reserved, label='cuda.memory_reserved')

    # Настройка заголовка и меток осей
    ax.set_title('GPU graph memory')
    ax.set_xlabel('Per Epoch')
    ax.set_ylabel('Memory')

    # Добавление легенды
    ax.legend()
    plt.grid()
    # Отображение графика
    plt.show()


def plt_cpu_use_graph(cpu):
    fig, ax = plt.subplots()

    # Построение графика для g_loss

    ax.plot(range(len(cpu)), cpu, label=['cpu1', 'cpu2', 'cpu3', 'cpu4'])

    # Настройка заголовка и меток осей
    ax.set_title('CPU graph usage')
    ax.set_xlabel('Per Epoch')
    ax.set_ylabel('Percent')

    # Добавление легенды
    ax.legend()
    plt.grid()
    # Отображение графика
    plt.show()
