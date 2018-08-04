#coding=utf-8
import numpy as np
import os
from collections import Counter
import librosa
import time

# 获得训练用的wav文件路径列表
def get_wave_files(wav_path):
    wav_files = []
    total_count = 0
    for (dirpath,dirnames,filenames) in os.walk(wav_path):#访问文件夹下的所有文件
    #os.walk() 方法用于通过在目录树种游走输出在目录中的文件名，向上或者向下
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                #endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
                filename_path = os.sep.join([dirpath,filename])#定义文件路径(连)
                if os.stat(filename_path).st_size < 240000:#st_size文件的大小，以位为单位
                    total_count += 1
                    continue
                wav_files.append(filename_path)#加载文件
    return wav_files

#读取wav文件对应的label
def get_wav_label(label_file, wav_files):
    labels_dict = {}
    with open(label_file, 'r') as f:
        for label in f :
            label =label.strip('\n')
            label_id = label.split(' ',1)[0]
            label_text = label.split(' ',1)[1]
            labels_dict[label_id]=label_text#以字典格式保存相应内容
    labels=[]
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        #得到相应的文件名后进行'.'分割
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])#存在该标签则放入
            new_wav_files.append(wav_file)

    return new_wav_files,labels#返回标签和对应的文件

class AudioBatch(object):
    def __init__(self, folder='train/', text='train/word.txt', max_input = 512, max_label = 256):
        self.pointer    = 0
        self.batch_size = 16#每次取16个文件
        self.mfcc_batches  = []
        self.label_batches = []
        self.mfcc_max_len   = max_input
        self.label_max_len  = max_label
        wav_files = get_wave_files(folder)#获取文件名列表
        self.n_batch = len(wav_files)//self.batch_size
        files,labels = get_wav_label(text, wav_files)#得到标签和对应的语音文件
        print "加载训练样本:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print "样本数:", len(files)

        #词汇表（参考对话、诗词生成）
        all_words = []
        for label in labels:
            all_words += [word for word in label]
        counter = Counter(all_words)
        count_pairs = sorted(counter.items(),key=lambda x: -x[1])
        words,_= zip(*count_pairs)
        self.words_size = len(words)#词汇表尺寸
        print '词汇表大小:', self.words_size

        #词汇映射成id表示
        word_num_map = dict(zip(words, range(len(words))))
        to_num = lambda word: word_num_map.get(word,len(words))#词汇映射函数
        labels_vector = [list(map(to_num,label)) for label in labels]
        label_max_len = np.max([len(label) for label in labels_vector])#获取最长字数
        print '最长句子的字数:', label_max_len

        mfcc_max_len = 0
        for i in range(len(files)):
            wav, sr = librosa.load(files[i], mono=True)#处理语音信号的库librosa
            #加载音频文件作为a floating point time series.（可以是wav,mp3等格式）mono=True：signal->mono
            mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])#转置特征参数
            if len(mfcc) > mfcc_max_len :
                mfcc_max_len = len(mfcc)
            if len(mfcc) > self.mfcc_max_len:
                print "... mfcc  超过最大长度：", len(mfcc), self.mfcc_max_len
                continue
            if len(labels_vector[i]) > self.label_max_len:
                print "... label 超过最大长度：", len(labels_vector[i]), self.label_max_len
                continue
            self.mfcc_batches.append(mfcc.tolist())
            self.label_batches.append(labels_vector[i])
            #librosa.feature.mfcc特征提取函数
        print "最长的语音:", mfcc_max_len

    def get_batches(self, batch_size):
        mfcc_batch  = []
        label_batch = []
        for i in range(batch_size):
            mfcc_batch.append(self.mfcc_batches[self.pointer])
            label_batch.append(self.label_batches[self.pointer])
            self.pointer += 1
        for mfcc in mfcc_batch:
            while len(mfcc) < self.mfcc_max_len:
                mfcc.append([0]*20)#补一个全0列表
        for label in label_batch:
            while len(label) < self.label_max_len:
                label.append(0)
        return mfcc_batch, label_batch

    def update_batches(self):
        self.pointer = 0

    def get_n_batch(self):
        return self.n_batch

    def get_word_size(self):
        return self.words_size
