import matplotlib.pyplot as plt
import numpy as np

# References
# https://blogs.zhinst.com/mehdia/2019/11/25/how-to-generate-qam-signaling-with-the-hdawg-arbitrary-waveform-generator/

class Defines:

    # Sampling frequency
    FS = 100E9
    # Number of samples
    N_SAMPLES = 1000

def export_as_pwl(name, time, amplitude):
    file = open(name,"w")

    for local_index in range(0, len(amplitude)):
        file.write(str(time[local_index]))
        file.write(" ")
        file.write(str(amplitude[local_index]))
        file.write("\n")
    file.close()

def generate_QAM_signal(carrier_frequency, depth, symbol_duration):

    [i_t, q_t] = generate_QAM_symbols(depth, symbol_duration)

    [position, sin_t] = generate_sine_wave(carrier_frequency, 0, 0, 1, len(i_t))
    [position, cos_t] = generate_sine_wave(carrier_frequency, np.pi / 2, 0, 1, len(q_t))

    time = position / Defines.FS
    print("Duration of the signal:", str((time[-1] - time[0])*1000), "ms")
    s_t = i_t * sin_t + q_t * cos_t

    return [time, s_t]

def generate_QAM_symbols(depth, symbol_duration):

    M = depth
    x = np.ones(int(symbol_duration*Defines.FS))
    wi = np.ones(int(symbol_duration*Defines.FS))
    wq = np.ones(int(symbol_duration*Defines.FS))

    wiAcc = []
    wqAcc = []

    for i in range(-(M - 1), M, 2):
        for q in range(-(M - 1), M, 2):
            wi = (i / M) * x
            wq = (q / M) * x
            wiAcc = [*wiAcc, *wi]
            wqAcc = [*wqAcc, *wq]

    return [wiAcc, wqAcc]

# Generates a sine wave
# Inputs: frequency (Hz), phase (rad/s), offset (V), amplitude (V), size (amount of samples)
def generate_sine_wave(frequency, phase, offset, amplitude, size):
    local_frequency = frequency
    DC = offset
    AMP = amplitude

    w = 2 * np.pi * local_frequency
    x = np.arange(size)

    y = DC + AMP*np.sin(w * x / Defines.FS + phase)

    return [x, y]

def main():
    print("Aloha!")
    return True

if __name__ == "__main__":

    carrier_frequency = 2.4E9
    symbol_duration = 3.6/(1E6)

    [time, s_t] = generate_QAM_signal(carrier_frequency, 4, symbol_duration)
    plt.plot(time, s_t)

    export_as_pwl("modulated.txt", time, s_t)

    plt.show()
