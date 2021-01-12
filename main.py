import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import rflib as rf

# References
# https://blogs.zhinst.com/mehdia/2019/11/25/how-to-generate-qam-signaling-with-the-hdawg-arbitrary-waveform-generator/

class Defines:

    # Sampling frequency
    FS = 100*10E9

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
def channel_model(vin, tau, noise_level, skew_level, describe_channel):

    if skew_level != 0:
        level = 0
        previous_level = 0
        for i in range(len(vin)):
            level = vin[i]
            if i == 0:
                previous_level = level

            if level != previous_level: #add random skew
                skew_index = np.random.randint(skew_level)
                skew_position = np.random.randint(2)

                if skew_position == 1:
                    for j in range(i-skew_index, i):
                        vin[j] = vin[i]
                elif skew_position == 0:
                    for j in range(i, i+skew_index):
                        vin[j] = vin[i-1]

            previous_level = level

    if noise_level != 0: #add random noise
        for i in range(len(vin)):
            vin[i] += np.random.uniform(-noise_level, noise_level)

    t = np.arange(1, (len(vin) + 1))

    ri = (1/(tau*Defines.FS)) * np.exp(- t / (tau*Defines.FS))
    vout = np.convolve(ri, vin)
    vout = vout[0:len(vin)]

    w = 10**np.arange(-1, 15, 0.1)
    alfa = 1/tau
    if describe_channel is True:
        my_complex = alfa + 1j*w
        tf = alfa/my_complex
        mag = abs(tf)
        phase = np.arctan(tf.imag/tf.real)
        phase = rf.rad_to_degree(phase)

        #plt.figure()
        #plt.semilogx(w, mag, label="magnitude")
        #plt.legend(loc="upper left")
        #plt.grid(True)
        #plt.show()

        #plt.figure()
        #plt.semilogx(w, phase, label="phase")
        #plt.legend(loc="upper left")
        #plt.grid(True)
        #plt.show()

        return [mag, phase, w, vout, ri]


    return [vout, ri]

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

def generate_eye_diagram(data, BIT_DURATION_SAMPLES):

    buffer = np.zeros((int(len(data)/BIT_DURATION_SAMPLES), BIT_DURATION_SAMPLES))
    eye = np.zeros((int(len(data) / BIT_DURATION_SAMPLES), 3*BIT_DURATION_SAMPLES))

    j = 0

    for i in range(0, len(data), BIT_DURATION_SAMPLES):
        buffer[j, 0:BIT_DURATION_SAMPLES] = data[i:BIT_DURATION_SAMPLES+j*BIT_DURATION_SAMPLES]
        bit_value = buffer[j, BIT_DURATION_SAMPLES-1] #np.ceil(np.average(buffer[j, 0:BIT_DURATION_SAMPLES]))

        if j < 2:
            eye[j, 0 * BIT_DURATION_SAMPLES:1 * BIT_DURATION_SAMPLES] = buffer[j, 0:BIT_DURATION_SAMPLES]
            eye[j, 1 * BIT_DURATION_SAMPLES:2 * BIT_DURATION_SAMPLES] = buffer[j, 0:BIT_DURATION_SAMPLES]
            eye[j, 2 * BIT_DURATION_SAMPLES:3 * BIT_DURATION_SAMPLES] = buffer[j, 0:BIT_DURATION_SAMPLES]
        else:
            eye[j, 0 * BIT_DURATION_SAMPLES:1 * BIT_DURATION_SAMPLES] = buffer[j-2, 0:BIT_DURATION_SAMPLES]
            eye[j, 1 * BIT_DURATION_SAMPLES:2 * BIT_DURATION_SAMPLES] = buffer[j-1, 0:BIT_DURATION_SAMPLES]
            eye[j, 2 * BIT_DURATION_SAMPLES:3 * BIT_DURATION_SAMPLES] = buffer[j, 0:BIT_DURATION_SAMPLES]

        j = j + 1

    return eye

def continous_time_linear_equalizer(vin, rd, gm, rs, cs, describe_CTLE):

    #TODO add the c_band_stop capacitor, adjust the timing with Defines.FS
    c_band_stop = 1E-12

    t = np.arange(1, (len(vin) + 1))

    alfa = - gm * rd
    tau = rs * cs * Defines.FS
    a = 1/tau
    beta = gm * rs / 2
    b = (beta + 1)/tau

    ri = alfa * ((a - b)*np.exp(-b*t/1))

    vout = np.convolve(ri, vin)
    vout = vout[0:len(vin)]

    w = 10 ** np.arange(-1, 15, 0.1)
    alfa = -gm * rd
    tau = rs * cs * Defines.FS
    beta = gm * rs / 2
    if describe_CTLE is True:
        my_complex = 1 + 1j * w * tau
        tf = alfa / (1 + beta*(1/my_complex))
        mag = abs(tf)
        phase = np.arctan(tf.imag / tf.real)
        phase = rf.rad_to_degree(phase)

        #plt.semilogx(w, mag, label="mag")
        #plt.legend(loc="upper left")
        #plt.grid(True)
        #plt.show()

        #plt.semilogx(w, phase, label="phase")
        #plt.legend(loc="upper left")
        #plt.grid(True)
        #plt.show()

        return [mag, phase, w, vout, ri]

    return [vout, ri]

def feed_forward_equalizer(vin, amplitude_precursor, amplitude_poscursor, duration_precursor, duration_poscursor):

    vout = np.zeros(len(vin))
    previous_bit = 0
    i = 0

    while i < len(vin):
        present_bit = vin[i]

        if present_bit > previous_bit:
            # It is a rising edge
            vout[i - duration_poscursor:i] = amplitude_poscursor

            vout[i:i + duration_precursor] = present_bit + amplitude_precursor
            i += duration_precursor - 1

        elif present_bit < previous_bit:
            # It is a rising edge
            vout[i - duration_precursor:i] = previous_bit - amplitude_poscursor

            vout[i:i + duration_poscursor] = -amplitude_precursor
            i += duration_poscursor - 1

        else:
            vout[i] = present_bit

        previous_bit = present_bit
        i += 1

    return vout


# Generates a sequence of bits randomically or not.
def generate_bitstream(sequence_size, bit_duration_samples, is_random):

    if is_random is False:
        # numpy.tiles repeats the sequence in the input array
        sequence = [1, 0]
        pattern = np.tile(sequence, int(sequence_size/2))

    elif is_random is True:
        # numpy.random.randint generates a sequence of size "size" and varying from 0 to 1
        pattern = np.random.randint(2, size=sequence_size)

    # numpy.repeat oversamples by bit_duration_samples the "pattern" array
    stream = np.repeat(pattern, bit_duration_samples)

    return stream

if __name__ == "__main__":

    carrier_frequency = 2.4E9
    symbol_duration = 3.6/(1E6)

    BIT_DURATION_SECONDS = 100E-12 #100E-12
    BW = 1/BIT_DURATION_SECONDS
    print(str(BW/1E9) + " GHz")
    BIT_DURATION_SAMPLES = int(BIT_DURATION_SECONDS*Defines.FS)
    SEQUENCE_SIZE = 4
    sequence = generate_bitstream(SEQUENCE_SIZE, BIT_DURATION_SAMPLES, False)
    plt.figure()
    plt.plot(sequence, label="sequence")

    AMP_PRE = 0.3
    AMP_POS = -AMP_PRE
    DUR_PRE = 8
    DUR_POS = 8
    tx_out = feed_forward_equalizer(sequence, AMP_PRE, AMP_POS, DUR_PRE, DUR_POS)
    plt.plot(tx_out, label="tx_out")
    plt.legend(loc="lower right")
    plt.title("TX OUT data (with and without FFE)")
    plt.show(block=False)

    NOISE_LEVEL = 0.3
    SKEW_LEVEL = int(BIT_DURATION_SAMPLES/8)
    TAU = 4E-11
    DESCRIBE_MODEL = True

    if DESCRIBE_MODEL is True:
        [magc, phase, w, rx_in, ri] = channel_model(tx_out, TAU, NOISE_LEVEL, SKEW_LEVEL, DESCRIBE_MODEL)
    else:
        [rx_in, ri] = channel_model(tx_out, TAU, NOISE_LEVEL, SKEW_LEVEL, DESCRIBE_MODEL)

    plt.figure()
    plt.plot(rx_in, label="rx_in")
    plt.legend(loc="lower right")
    plt.title("RX IN channel only")
    plt.show(block=False)

    RD = 100
    GM = 0.6
    RS = 10
    CS = 5E-12

    DESCRIBE_MODEL = True
    if DESCRIBE_MODEL is True:
        [magct, phase, w, eq_out, ri] = continous_time_linear_equalizer(tx_out, RD, GM, RS, CS, DESCRIBE_MODEL)
    else:
        [eq_out, ri] = continous_time_linear_equalizer(tx_out, RD, GM, RS, CS, DESCRIBE_MODEL)

    plt.figure()
    plt.plot(eq_out, label="eq_out")
    plt.legend(loc="lower right")
    plt.title("RX IN channel + CTLE")
    plt.show(block=False)

    plt.figure()
    plt.semilogx(w, magct, label="channel + CTLE")
    plt.semilogx(w, magc, label="channel")
    plt.legend(loc="lower left")
    plt.title("Frequency responses")
    plt.show(block=False)

    eye = generate_eye_diagram(rx_in, BIT_DURATION_SAMPLES)
    plt.figure()
    for i in range(0, len(eye)):
        plt.plot(eye[i,:])
    plt.ylim(-0.5, 1.5)
    plt.title("RX IN channel")
    plt.show(block=False)

    eye = generate_eye_diagram(eq_out, BIT_DURATION_SAMPLES)
    plt.figure()
    for i in range(0, len(eye)):
        plt.plot(eye[i,:])
    #plt.ylim(-0.5, 1.5)
    plt.title("RX IN channel + CTLE ")
    plt.show(block=False)

    plt.show()



    #
    #plt.show()







    #[frequency, amplitude] = fft_of(ri)
    #plt.semilogx(frequency, amplitude)

    #plt.show()





