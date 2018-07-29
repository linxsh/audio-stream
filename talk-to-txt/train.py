#coding:utf-8
import tensorflow as tf
import numpy as np
from model import AudioResNet

train_folder = './train'
train_words  = './train/word.txt'

if __name__ == '__main__':
    model = AudioResNet()
    model.train(folder = train_folder, text = train_words, batch_size = 16)
