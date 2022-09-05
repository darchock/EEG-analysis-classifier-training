function [std_ps_data, std_ps_left, std_ps_right] = compute_ps_std(data, left_data, right_data, time_seg, ...
          freq, which_channel, window, overlap, fs)
    % func purpose - computes STD of Power Spectrum in specified frequency
    %                band and time segment, to be features for our
    %                Classifer and to be displayed on Features Histograms Graph
    % @ input: data = training_data of all 128 trials
    %          left_data = data of tagged left hand trials only
    %          right_data = data of tagged right hand trials only
    %          time_seg = time segment on which we calculate the Power Spectrum on
    %          freq = frequency range on which we calculate the Power Spectrum on
    %          which_channel = from which channel C3\C4 we calculate the Power Spectrum on
    %          window = length of window time
    %          overlap = size of overlap between windows
    %          fs = sampling rate in [Hz]
    % @ output: std_ps_data = STD of Power Spectrum on all 128 trials to be
    %                         a feature for our Classifier
    %           std_ps_left\right = STD of Power Spectrum on tagged
    %                               left\right trials to be displayed on Histograms

    left_ps      = pwelch(left_data(:, time_seg, which_channel)', window, overlap, freq, fs);
    right_ps     = pwelch(right_data(:, time_seg, which_channel)', window, overlap, freq, fs);
    data_ps      = pwelch(data(:, time_seg, which_channel)', window, overlap, freq, fs);

    std_ps_left  = std(left_ps);
    std_ps_right = std(right_ps);
    std_ps_data  = std(data_ps);

end