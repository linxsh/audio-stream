from pyaudio import PyAudio,paInt16
import numpy as np
import wave

framerate=16000
channel=1
samplewidth=2
samplepoints=400
sampleshifts=160

class AudioRecord(object):
    def __init__(self):
        pa=PyAudio()
        self.stream=pa.open(format = paInt16, channels = channel, rate=framerate, input=True, frames_per_buffer=samplepoints)
        self.save_audio_array=np.zeros(0, dtype=np.short)

    def read_frame(self):
        if np.size(self.save_audio_array) < samplepoints:
            audio_string=self.stream.read(samplepoints)
            audio_array=np.fromstring(audio_string, dtype=np.short)
            self.save_audio_array=np.append(self.save_audio_array, audio_array)
            #print("read %d"%(np.size(audio_array)))
        read_audio_array = self.save_audio_array[:samplepoints]
        self.save_audio_array=self.save_audio_array[sampleshifts:]
        #print("write %d, res: %d"%(np.size(read_audio_array), np.size(self.save_audio_array)))
        return read_audio_array

    def save_to_file(self, filename=None, data=np.zeros(0, dtype=np.short)):
        if filename is not None:
            wavefile=wave.open(filename,'wb')
            wavefile.setnchannels(channel)
            wavefile.setsampwidth(samplewidth)
            wavefile.setframerate(framerate)
            wavefile.writeframes(b"".join(data.tostring()))
            wavefile.close()
