import numpy as np
import matplotlib.pyplot as plotter
import scipy.signal as signal

def compute_fishel_spectral_slip(pac, show_plots=False):
    '''
    pac - time series of AC pressure values
    '''

    num_samples_per_frame = 22
    num_frames_per_feat = 1
    n_fft = num_samples_per_frame*num_frames_per_feat
    sampling_freq = 2200
    nyq_rate = sampling_freq/2.0
    high_pass_cutoff_freq = 5

    # TODO: Highpass filter the PAC signal
    Wp = high_pass_cutoff_freq / nyq_rate   # Cutoff frequency
    b,a = signal.iirfilter(1, Wp, btype='highpass', ftype='butter')
    pac_high_pass = signal.lfilter(b, a, pac)



    # Band pass of spectrogram
    freq_min = 30
    freq_max = 200

    if show_plots:
        plotter.subplot(3,1,1)

        plotter.plot(pac_high_pass)

        plotter.subplot(3,1,2)
    (Pxx, freqs, t, handle) = plotter.specgram(pac_high_pass, NFFT = n_fft, Fs = sampling_freq, sides='onesided',
                                               scale_by_freq=False, noverlap = 0)
    p_filtered, filter_energy =  bandpass_filter_response(Pxx, freqs, freq_min, freq_max)
    if show_plots:

        plotter.subplot(3,1,3)
        plotter.plot(t, filter_energy)

        plotter.suptitle('Fishel Spectral Slip Detection')
        plotter.show()
    return (t, filter_energy)

def bandpass_filter_response(Pxx, freqs, freq_min, freq_max):
    min_idx = 0
    max_idx = len(freqs)
    max_found = False
    for i, f in enumerate(freqs):
        if f < freq_min:
            min_idx += 1
        if f > freq_max and not max_found:
            max_idx = i
            max_found = True


    p_filtered = Pxx[range(min_idx, max_idx), :]

    filter_energy = np.sum(p_filtered, axis=0)

    return (p_filtered, filter_energy)
