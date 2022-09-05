function [data_se_mean, time_se] = compute_se(data, start_time, n_trials, fs, which_channel)
    % func purpose - computes Spectral Entropy of the EEG signal as a
    % measure of its spectral power distribution
    % @ input: data = left\right data specified for tagged left\right
    %                 trials or the whole 128 trials training_data
    %          start_time = start time for the Spectral Entropy calculation
    %          n_trials = num of trials depending on the data given
    %          fs = sampling rate in [Hz]
    %          which_channel = from which Channel the data is extracted
    % @ output: data_se_mean = vector of mean spectral power distribution
    %                          of each trial
    %           time_se = time vector of the Spectral Entropy
    
    data_se_mean = zeros(n_trials, 1);
    for tr = 1 : n_trials
        [data_se, time_se] = pentropy(data(tr, start_time : end, which_channel), fs);
        data_se_mean(tr) = mean(data_se);
    end

end