function [left_std, right_std, data_std] = compute_std(data, left_data, right_data, time_seg, which_channel)
    % func purpose - computing STD in EEG signal to be features for our Classifier
    % @ input: data = training_data of all 128 trials
    %          left_data = data of tagged left hand trials only
    %          right_data = data of tagged right hand trials only
    %          time_seg = time segment from which we extract the EEG signal from
    %          which_channel = from which channel C3\C4 we extract the
    %                          signal from

    left_std = std(left_data(:, time_seg, which_channel), [], 2);
    right_std = std(right_data(:, time_seg, which_channel), [], 2);
    data_std = std(data(:, time_seg, which_channel), [], 2);

end