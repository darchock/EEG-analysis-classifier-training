function [RT_left_c3, RT_right_c3, RT_data_c3, RT_left_c4, RT_right_c4, RT_data_c4] = compute_rtp(data, ...
          left_data, right_data, RTP_time, window, overlap, freq, fs)
    % func purpose - computes Root Total Power - square root of Power
    % Spectrum sum.
    % @ input: data = training_data of all 128 trials
    %          left_data = data of only tagged left hand trials
    %          right_data = data of only tagged right hand trials
    %          RTP_time = time segment of which we computed the RTP
    %          window = length of window - specified for RTP calculation
    %          overlap = size of overlap between windows - specified for RTP calculation
    %          freq = frequencies spectrum in [Hz]
    %          fs = sampling rate in [Hz]
    % @ ouput: RT_left\right_c3\c4 = Root Total Power of each hand and each Channel
    %          RT_data_c3\c4 = Root Total Power of all 128 trials to be a
    %          feature for the Classifier.

    % Left hand
    pw_left_c3  = pwelch(left_data(:, RTP_time, 1)', window, overlap, freq, fs);
    pw_right_c3 =  pwelch(right_data(:, RTP_time, 1)', window, overlap, freq, fs);
    data_c3     = pwelch(data(:, RTP_time, 1)', window, overlap, freq, fs);
    RT_left_c3  = sqrt(sum(pw_left_c3));
    RT_right_c3 = sqrt(sum(pw_right_c3));
    RT_data_c3  = sqrt(sum(data_c3));
    
    % Right hand
    pw_left_c4  = pwelch(left_data(:, RTP_time, 2)', window, overlap, freq, fs);
    pw_right_c4 =  pwelch(right_data(:, RTP_time, 2)', window, overlap, freq, fs);
    data_c4     = pwelch(data(:, RTP_time, 2)', window, overlap, freq, fs);
    RT_left_c4  = sqrt(sum(pw_left_c4));
    RT_right_c4 = sqrt(sum(pw_right_c4));
    RT_data_c4  = sqrt(sum(data_c4)); 

end