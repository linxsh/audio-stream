import numpy as np

short_value = 32768.0
low_energy_gate  = 1.0e-5
low_energy_frame_cnt = 60
low_energy_weight = 5
high_energy_gate = 1.0e-4
high_energy_time = 10
store_frame_cnt = 100

def vad_energy(frame_array):
    if np.size(frame_array) > 0:
        normalize_data = frame_array / short_value
        energy = np.mean(normalize_data ** 2)
    else:
        energy = 0.0
    return energy

def update_energy_gate(energy_array, energy_level):
    update_low_energy_gate  = 0.0
    update_high_energy_gate = 0.0
    update = 0
    if energy_level[0] == 0 :
        low_energy_cnt = 1
        for i in range(1, np.size(energy_level)) :
            if energy_level[i] == 0:
                low_energy_cnt += 1
                if low_energy_cnt >= low_energy_frame_cnt :
                    update = 1
                    update_low_energy_gate  = np.mean(energy_array[:low_energy_frame_cnt])
                    update_low_energy_gate *= low_energy_weight
                    update_high_energy_gate = update_low_energy_gate * high_energy_time
                    #print "=====>", update_low_energy_gate, update_high_energy_gate
                    break
            else :
                break
    return update, update_low_energy_gate, update_high_energy_gate

def get_active_state(active, energy_level) :
    active_state = 0
    if active == 0 :
        active_score = 0
        if energy_level[-1] == 2 :
            frame_cnt = 0
            for i in range(np.size(energy_level) - 1, 1, -1) :
                if energy_level[i] == 2 :
                    active_score += 2
                if energy_level[i] == 1 :
                    active_score += 1
                if energy_level[i] == 0:
                    break
                frame_cnt += 1
                if frame_cnt >= 4 :
                    break
        if active_score >= 7 :
            active_state = 1
    else :
        active_score = 60
        if energy_level[-1] != 2 :
            frame_cnt = 0
            for i in range(np.size(energy_level) - 1, 1, -1) :
                if energy_level[i] == 1 :
                    active_score -= 1
                if energy_level[i] == 0 :
                    active_score -= 2
                frame_cnt += 1
                if frame_cnt >= 30 :
                    break
        if active_score >= 10:
            active_state = 1
    return active_state


#def vad_zero_rate(frame_array):

class AudioVad(object):
    def __init__(self, sample_rate = 16000, sample_time = 0.025, sample_shift = 0.010):
        self.audio_array   = np.zeros(0, dtype=np.short)
        self.energy_array  = np.zeros(0, dtype=np.float)
        self.energy_level  = np.zeros(0, dtype=np.short)
        self.active_state  = 0
        self.low_energy_gate  = low_energy_gate
        self.high_energy_gate = high_energy_gate
        self.sample_length = int(round(sample_rate * sample_time))
        self.sample_shifts = int(round(sample_rate * sample_shift))

    def write_frame(self, audio_input_array) :
        self.audio_array  = np.append(self.audio_array, audio_input_array)
        while np.size(self.audio_array) >= self.sample_length :
            frame_array = self.audio_array[:self.sample_length]
            self.audio_array = self.audio_array[self.sample_shifts : ]
            energy = vad_energy(frame_array)
            self.energy_array = np.append(self.energy_array, energy)
            if energy >= self.low_energy_gate:
                if energy >= self.high_energy_gate:
                    self.energy_level = np.append(self.energy_level, 2)
                else:
                    self.energy_level = np.append(self.energy_level, 1)
            else:
                self.energy_level = np.append(self.energy_level, 0)
            if np.size(self.energy_array) > store_frame_cnt :
                self.energy_array = np.delete(self.energy_array, 0, axis = 0)
            if np.size(self.energy_level) > store_frame_cnt :
                self.energy_level = np.delete(self.energy_level, 0, axis = 0)
            self.active_state = get_active_state(self.active_state, self.energy_level)
        update, update_low_energy_gate, update_high_energy_gate = update_energy_gate(self.energy_array, self.energy_level)
        if update == 1 :
            self.low_energy_gate = update_low_energy_gate
            self.high_energy_gate = update_high_energy_gate
        #if self.active_state == 1 :
        #    print self.energy_level, self.active_state, self.low_energy_gate, self.high_energy_gate

    def get_vad_state(self) :
        return self.active_state
