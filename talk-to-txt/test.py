#coding:utf-8
import time
from datetime import datetime
from model import AudioResNet
from batches import AudioBatch

if __name__ == '__main__':
    audio_batch = AudioBatch(folder='train/', text='train/word.txt', batch_size = 1)
    audio_batch.update_batches()
    model = AudioResNet()
    start_time = datetime.now()
    model.load()
    print "加载时长: %s" % (datetime.now() - start_time)
    for batch in range(audio_batch.get_n_batch()):
        start_time = datetime.now()
        batches_wavs, batches_labels = audio_batch.get_batches()
        model.sample(batches_wavs)
        print "运算时长: %s" % (datetime.now() - start_time)
