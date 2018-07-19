from record import AudioRecord
from vad import AudioVad
import numpy as np

if __name__ == '__main__':
    arec=AudioRecord()
    avd=AudioVad()
    while True:
        data=arec.read_frame()
        avd.write_frame(data)
        avd_data, avd_state=avd.read_frame()
