from pyaudio import PyAudio,paInt16
import numpy as np

class AudioRecord(object):
    def __init__(self, sample_rate = 16000, sample_channel = 1, sample_width = 2, sample_time = 0.025):
        samplebuffer=int(round(sample_rate * sample_channel * sample_width * sample_time))
        pa=PyAudio()
        self.stream=pa.open(format = paInt16, channels=sample_channel, rate=sample_rate, input=True,
            frames_per_buffer=samplebuffer)
        self.samplepoints=int(round(sample_rate*sample_time))

    def read_frame(self):
        audio_string=self.stream.read(self.samplepoints)
        audio_array=np.fromstring(audio_string, dtype=np.short)
        return audio_array

    def write_frame(self, audio_array):
        audio_string=audio_array.to_string()
        self.stream.write(audio_string)
