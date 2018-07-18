from pyaudio import PyAudio,paInt16
import numpy as np
import wave

framerate=8000
framelength=400
frameshift=400
channel=1
samplewidth=2
samplepoints=200

class AudioRecord(object):
    def __init__(self, filename=None):
        pa=PyAudio()
        self.stream=pa.open(format = paInt16, channels = channel, rate=framerate, input=True, frames_per_buffer=framelength)
        self.pabuf=[]
        if filename is not None:
            self.wavefile=wave.open(filename,'wb')
            self.wavefile.setnchannels(channel)
            self.wavefile.setsampwidth(samplewidth)
            self.wavefile.setframerate(framerate)

    def read_frame(self):
        if len(self.pabuf) < framelength:
          audio_data=self.stream.read(samplepoints)
          self.pabuf.extend(audio_data)
        rbuf=self.pabuf[:framelength]
        del self.pabuf[:frameshift]
        return np.array(rbuf)

    def save_to_file(self, data):
        self.wavefile.writeframes(b"".join(data.tolist()))
