import numpy as np
import wave

sample_time=0.025 ##ms

class FileOp:
    R=0
    W=1

class AudioFile(object):
    def __init__(self, filename, mode = FileOp.W, sample_time = 0.025):
        if mode == FileOp.W:
            self.wavefile = wave.open(filename, 'wb')
            self.sample_rate    = 16000
            self.sample_channel = 1
            self.sample_width   = 2
            self.sample_time    = sample_time
        if mode == FileOp.R:
            self.wavefile = wave.open(filename, 'rb')
            params = self.wavefile.getparams()
            self.sample_rate    = params[0]
            self.sample_channel = params[1]
            self.sample_width   = params[2]

    def get_info(self):
        return self.sample_rate, self.sample_channel, self.sample_width

    def set_info(self, sample_rate, sample_channel, sample_width):
        self.sample_rate, self.sample_channel, self.sample_width = sample_rate, sample_channel, sample_width
        self.wavefile.setframerate(sample_rate)
        self.wavefile.setnchannels(sample_channel)
        self.wavefile.setsampwidth(sample_width)

    def write_frame(self, audio_array):
        self.wavefile.writeframes(b"".join(audio_array.tostring()))

    def read_frame(self):
        sample_length = int(round(self.sample_rate * self.sample_time))
        audio_string  = self.wavefile.readframes(sample_length)
        audio_array   = np.fromstring(audio_string, dtype=np.short)
        return audio_array
