#coding:utf-8
import tensorflow as tf
from model import AudioResNet
import librosa
from batches import get_wave_files

wav_path = './test'
if __name__ == '__main__':
    wav_files = get_wave_files(wav_path)
    for wav_file in wav_files:
        wav, sr = librosa.load(wav_file, mono=True)
        mfcc = librosa.feature.mfcc(wav, sr)
        model = AudioResNet()
        model.sample(mfcc)
