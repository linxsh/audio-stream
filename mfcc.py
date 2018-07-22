#coding: utf-8
import numpy as np
from scipy.fftpack import dct

pre_emphasis = 0.97
fft_length = 512
mel_filter_size = 40
mfcc_cpes_num = 12
log_fbank_count = mfcc_count = 20

class AudioMFCC(object):
    def __init__(self, sample_rate = 16000, sample_time = 0.025, sample_shift = 0.010):
        self.audio_sample_point = 0
        self.sample_rate = sample_rate
        self.sample_length = int(round(sample_rate * sample_time))
        self.sample_shifts = int(round(sample_rate * sample_shift))
        self.audio_array = np.zeros(0, dtype=np.short)
        self.log_fbank = np.zeros((0, mel_filter_size), dtype=np.float)
        self.mfcc = np.zeros((0, mfcc_cpes_num), dtype=np.float)
        #mel滤波器
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, mel_filter_size + 2)
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))
        hz_bin = np.floor((fft_length + 1) * hz_points / sample_rate)
        self.fbank_filter = np.zeros((mel_filter_size, int(np.floor(fft_length / 2 + 1))))
        for m in range(1, mel_filter_size + 1):
            f_m_minus = int(hz_bin[m - 1])
            f_m       = int(hz_bin[m])
            f_m_plus  = int(hz_bin[m + 1])
            for k in range(f_m_minus, f_m):
                self.fbank_filter[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                self.fbank_filter[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    def write_frame(self, audio_input_array):
        #预处理: y[n] = x[n] - pre_emphasis * x[n - 1]
        pre_emphasis_array = np.append(audio_input_array[0] - pre_emphasis * self.audio_sample_point,
            audio_input_array[1:] - pre_emphasis * audio_input_array[:-1])
        self.audio_sample_point = audio_input_array[-1]
        self.audio_array = np.append(self.audio_array, pre_emphasis_array)
        while np.size(self.audio_array) >= self.sample_length :
            #分帧
            frame_array = self.audio_array[:self.sample_length]
            frame_array = frame_array.astype(np.float)
            self.audio_array = self.audio_array[self.sample_shifts:]
            #加hamming窗: w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
            frame_array *= np.hamming(self.sample_length)
            #傅立叶变换和功率谱
            mag_array = np.absolute(np.fft.rfft(frame_array, fft_length))
            power_array = ((1.0 / fft_length) * ((mag_array) ** 2))
            #将频率转换为Mel
            fbank_arry = np.dot(power_array, self.fbank_filter.T)
            fbank_arry = np.where(fbank_arry == 0, np.finfo(float).eps, fbank_arry)
            log_fbank_arry = 20 * np.log10(fbank_arry)
            #print log_fbank_arry.shape
            #print log_fbank_arry.reshape(np.size(log_fbank_arry),-1).shape
            self.log_fbank = np.row_stack((self.log_fbank, log_fbank_arry))
            mfcc = dct(log_fbank_arry.reshape(np.size(log_fbank_arry),-1), type=2, axis=1, norm='ortho')
            #print mfcc.reshape(np.size(mfcc))[:mfcc_cpes_num].shape
            self.mfcc = np.row_stack((self.mfcc, mfcc.reshape(np.size(mfcc))[:mfcc_cpes_num]))

    def get_mffc(self):
        log_fbank = self.log_fbank
        self.log_fbank = np.zeros((0, mel_filter_size), dtype=np.float)
        mfcc = self.mfcc
        self.mfcc = np.zeros((0, mfcc_cpes_num), dtype=np.float)
        return log_fbank, mfcc


