#coding:utf-8
import tensorflow as tf
import numpy as np
from batches import AudioBatch

train_folder = './train'
train_words  = './train/word.txt'

if __name__ == '__main__':
    batches = AudioBatch(train_folder, train_words)
    mfcc_batch, label_batch = batches.get_batches(16)
    print mfcc_batch
    print label_batch
