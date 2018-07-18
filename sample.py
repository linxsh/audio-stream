from record import AudioRecord
import numpy as np

if __name__ == '__main__':
    arec=AudioRecord()
    while True:
        data=arec.read_frame()
