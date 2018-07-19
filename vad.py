import numpy as np

short_value=32768.0
low_energy_tolerance=0.1
high_energy_tolerance=0.5
vad_frame_cnt=20

def vad_energy(data):
    if np.size(data) > 0:
        normalize_data=data/short_value
        normalize_energy=np.dot(normalize_data, normalize_data.T)/np.size(normalize_data)
    else:
        normalize_energy = 0.0
    return normalize_energy

#def vad_zero_rate(data):

class AudioVad(object):
    def __init__(self):
        self.data_array=np.zeros((0,0), dtype=np.short)
        self.energy_array=np.zeros(0,   dtype=np.float)
        self.active_array=np.zeros(0,   dtype=np.short)

    def write_frame(self, data):
        energy = vad_energy(data)
        print energy
        if energy >= low_energy_tolerance:
            if energy >= high_energy_tolerance:
                self.data_array=np.row_stack(self.data_array, data)
                self.energy_array=np.append(slef.energy_array, 2)
            else:
                self.data_array=np.row_stack(self.data_array, data)
                self.energy_array=np.append(slef.energy_array, 1)
        else:
            self.data_array=np.row_stack(self.data_array, data)
            self.energy_array=np.append(slef.energy_array, 0)

        if np.size(self.energy_array) > 20:
            self.data_array=np.delete(self.data_array, 0, axis=0)
            self.energy_array=np.delete(self.energy_array, 0, axis=0)

    def read_frame(self):
        return self.data_array[-1], self.active_array[-1]
