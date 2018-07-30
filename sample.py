from stream import AudioStream
from fileop import AudioFile, FileOp
from vad import AudioVad
from mfcc import AudioMFCC
import numpy as np

sample_rate    = 8000
sample_channel = 1
sample_width   = 2
sample_time    = 0.025
sample_shift   = 0.010

if __name__ == '__main__':
    audio_rec  = AudioStream(sample_rate, sample_channel, sample_width, sample_time)
    audio_file = AudioFile('./a.wav', FileOp.W, sample_time)
    audio_file.set_info(sample_rate, sample_channel, sample_width)
    audio_vad  = AudioVad (sample_rate, sample_time, sample_shift)
    audio_mfcc = AudioMFCC(sample_rate, sample_time, sample_shift)
    frame_cnt = 0
    active_frame_cnt = 0
    unactive_frame_cnt = 0
    while True:
        audio_array = audio_rec.read_frame()
        audio_vad.write_frame(audio_array)
        frame_cnt += 1
        if audio_vad.get_vad_state() == 1:
            audio_mfcc.write_frame(audio_array)
            log_fbank, mfcc = audio_mfcc.get_mffc()
            audio_file.write_frame(audio_array)
            if (frame_cnt - active_frame_cnt) != 1 :
                print "active", active_frame_cnt, frame_cnt
            active_frame_cnt = frame_cnt
        else :
            audio_mfcc.update_mfcc()
            if (frame_cnt - unactive_frame_cnt) != 1 :
                print "unactive", unactive_frame_cnt, frame_cnt
            unactive_frame_cnt = frame_cnt
