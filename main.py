import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.fft import fft, fftfreq

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


# Reference: https://en.wikipedia.org/wiki/RC_circuit
def channel_model(vin, R, C, noise_level, skew_level):

    if skew_level != 0:
        level = 0
        previous_level = 0
        for i in range(len(vin)):
            level = vin[i]
            if i == 0:
                previous_level = level

            if level != previous_level: #add random skew
                skew_index = random.randint(skew_level)
                skew_position = random.randint(2)

                if skew_position == 1:
                    for j in range(i-skew_index, i):
                        vin[j] = vin[i]
                elif skew_position == 0:
                    for j in range(i, i+skew_index):
                        vin[j] = vin[i-1]

            previous_level = level

    if noise_level != 0: #add random noise
        for i in range(len(vin)):
            vin[i] += random.uniform(-noise_level, noise_level)

    t = np.arange(1,len(vin)+1)
    tau = R * C
    ri = (1/tau) * np.exp(- t / tau)
    vout = np.convolve(ri, vin)
    vout = vout[0:len(vin)]

    return vout

def generate_QAM_signal(carrier_frequency, depth, symbol_duration):

    [i_t, q_t] = generate_QAM_symbols(depth, symbol_duration)

    [position, sin_t] = generate_sine_wave(carrier_frequency, 0, 0, 1, len(i_t))
    [position, cos_t] = generate_sine_wave(carrier_frequency, np.pi / 2, 0, 1, len(q_t))

    time = position / Defines.FS
    print("Duration of the signal:", str((time[-1] - time[0])*1000), "ms")
    s_t = i_t * sin_t + q_t * cos_t

    return [time, s_t]


def fft_of(signal):

    n = len(signal)
    f_amplitude = fft(signal)
    frequency = fftfreq(n, 1/Defines.FS)[:n// 2]

    h_amplitude = 2.0 / n * np.abs(f_amplitude[0:n // 2])

    return [frequency, h_amplitude]

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
    w = 2 * np.pi * frequency
    x = np.arange(size)

    y = offset + amplitude*np.sin(w * x / Defines.FS + phase)

    return [x, y]

def generate_eye_diagram(data, bit_duration):

    buffer = np.zeros((int(len(data)/bit_duration), bit_duration))
    eye = np.zeros((int(len(data) / bit_duration), 3*bit_duration))

    j = 0

    for i in range(0, len(data), bit_duration):
        buffer[j, 0:bit_duration] = data[i:bit_duration+j*bit_duration]
        bit_value = buffer[j, bit_duration-1] #np.ceil(np.average(buffer[j, 0:bit_duration]))

        if j < 2:
            eye[j, 0 * bit_duration:1 * bit_duration] = buffer[j, 0:bit_duration]
            eye[j, 1 * bit_duration:2 * bit_duration] = buffer[j, 0:bit_duration]
            eye[j, 2 * bit_duration:3 * bit_duration] = buffer[j, 0:bit_duration]
        else:
            eye[j, 0 * bit_duration:1 * bit_duration] = buffer[j-2, 0:bit_duration]
            eye[j, 1 * bit_duration:2 * bit_duration] = buffer[j-1, 0:bit_duration]
            eye[j, 2 * bit_duration:3 * bit_duration] = buffer[j, 0:bit_duration]

        j = j + 1

    return eye


def add_feed_forward_equalizer(vin, amplitude_precursor, amplitude_poscursor, duration_precursor, duration_poscursor):

    vout = np.zeros(len(vin))
    present_bit = 0
    previous_bit = 0

    vin = np.array(vin)

    i = 0

    while i < len(vin):
        present_bit = vin[i]

        if present_bit > previous_bit:
            # It is a rising edge, thus, add the precursor
            vout[i - duration_poscursor:i] = amplitude_poscursor

            vout[i:i + duration_precursor] = present_bit + amplitude_precursor
            i += duration_precursor - 1

        elif present_bit < previous_bit:
            # It is a rising edge, thus, add the poscursor
            vout[i - duration_precursor:i] = previous_bit - amplitude_poscursor

            vout[i:i + duration_poscursor] = -amplitude_precursor
            i += duration_poscursor - 1




        else:
            vout[i] = present_bit

        previous_bit = present_bit
        i += 1

    return vout


# Generates a sequence of bits randomically or not.
# Inputs: size (size of the stream), bit_duration (number of samples each bit lasts),
# n_rise (number of samples of the rising edge), n_fall (number of samples of the falling edge),
# is_random (True to generate a random sequence, False to generate a repetitive 0/1 sequence).
# Output: sequence (the generated sequence)
def generate_bitstream(size, bit_duration, n_rise, n_fall, is_random):

    sequence = []
    bit_vector = np.zeros(int(bit_duration))

    if n_rise == 0:
        inc_rate = 1
    else:
        inc_rate = 1/n_rise

    if n_rise == 0:
        dec_rate = 1
    else:
        dec_rate = 1 / n_fall

    previous_bit = 0
    present_bit = 0
    j = 1

    fixed_pattern = np.zeros(int(size))
    for i in range(len(fixed_pattern)):
        fixed_pattern[i] = present_bit
        present_bit = not present_bit

    for k in range(size):

        if is_random is True:
            present_bit = random.randint(2)
        elif is_random is False:
            present_bit = fixed_pattern[k]

        if previous_bit == present_bit:
            bit_vector = np.ones(int(bit_duration)) * present_bit

        elif present_bit > previous_bit: #rising
            # TODO this section can be replaced by the pythonic way of indexing - see add_feed_forward_equalizer function
            for i in range(0, bit_duration):
                if i < n_rise:
                    bit_vector[i] = j*inc_rate
                    j += 1
                else:
                    bit_vector[i] = present_bit

        elif present_bit < previous_bit: #falling
            #TODO this section can be replaced by the pythonic way of indexing - see add_feed_forward_equalizer function
            for i in range(0, bit_duration):
                if i < n_fall:
                    bit_vector[i] = 1 - j*dec_rate
                    j += 1
                else:
                    bit_vector[i] = present_bit


        previous_bit = present_bit
        j = 1
        sequence = [*sequence, *bit_vector]

    return sequence

if __name__ == "__main__":

    carrier_frequency = 2.4E9
    symbol_duration = 3.6/(1E6)

    #[time, s_t] = generate_QAM_signal(carrier_frequency, 2, symbol_duration)
    #plt.plot(time, s_t)
    #plt.show()

    #[frequency, h_amplitude] = fft_of(s_t)

    #plt.plot(frequency, h_amplitude)
    #plt.show()


    #vin = np.concatenate([np.zeros(400), np.ones(200), np.zeros(4000)])
    #plt.plot(vin)
    #vout = channel_model(vin, 50, 10)

    BIT_DURATION = 20
    N_RISE = 0
    N_FALL = 0
    word = generate_bitstream(100, BIT_DURATION, N_RISE, N_FALL, False)
    plt.plot(word)


    AMP_PRE = 0.4
    AMP_POS = -0.2
    DUR_PRE = 6
    DUR_POS = 6
    tx_out = add_feed_forward_equalizer(word, AMP_PRE, AMP_POS, DUR_PRE, DUR_POS)
    plt.plot(tx_out)

    #tx_out = word

    NOISE_LEVEL = 0.2
    SKEW_LEVEL = BIT_DURATION/8
    rx_in = channel_model(tx_out, 3, 1, NOISE_LEVEL, SKEW_LEVEL)
    plt.plot(rx_in)
    plt.show()

    eye = generate_eye_diagram(rx_in, BIT_DURATION)
    for i in range(0, len(eye)):
        plt.plot(eye[i,:])
    plt.ylim(-0.3, 1.2)
    plt.show()




