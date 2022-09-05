function [spectro_mean, f, t] = compute_spec(data, which_channel, window, overlap, freq, fs, num_of_windows)
    % func purpose - computes Spectrogram using short-time Fourier transform according to the parameters
    % defined, convertes it to [dB] and then returns it's mean.
    % @ input: data = left\right_hand_matrix which is specified by only
    % their tagged trials
    %          which_channel = C3 or C4, window = length of window time
    %          overlap = size of overlap between windows
    %          freq = frequency spectrum in [Hz], 
    %          fs = frequency sampling rate in [Hz]
    %          num_of_windows = num of windows the data is divided into
    % @ output: spectro_mean = the mean of the spectrogram() output converted to [dB]
    % for each Hand and each Channel, saved into 'cell_spec'
    %           f = frequency spectrum vector
    %           t = vector of time instants

    n_trials  = size(data, 1);
    time_samp = length(freq);
    spectro   = zeros(n_trials, time_samp, num_of_windows);

    for sp = 1 : n_trials
        [s, f, t] = spectrogram(data(sp, :, which_channel), window, overlap, freq, fs, 'power', 'yaxis');
        spectro(sp, :, :) = abs(s) .^ 2;
    end

    % converting to db & calculating mean power
    spectro_in_db      = 10*log10(spectro);
    spectro_mean_in_db = mean(spectro_in_db, 1);
    spectro_mean       = squeeze(spectro_mean_in_db);
end