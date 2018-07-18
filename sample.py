from record import AudioRecord
import numpy as np

if __name__ == '__main__':
    arec=AudioRecord(filename="./a.wav")
    while True:
        arec.save_to_file(data=arec.read_frame())
