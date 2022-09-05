function [bandpower_left, bandpower_right, bandpower_feat] = compute_band(data, left_data, right_data, ...
                          fs, which_band, which_channel)
    % func purpose - computes using bandpower() and converts it to [dB]
    % for left\right trials only -- for Histogram plots, for all 128 trials
    % as features for Classification purposes.
    % @ input: data = the whole training_data
    %          left_data = data of only tagged left trials
    %          right_data = data of only tagged right trials
    %          fs = sampling rate in [Hz]
    %          which_band = two scalar of the frequency range of the band
    %          which_channel = from which Channel the band is extracted
    % @ output: bandpower_left = band Power in specified freq of left trials
    %           bandpower_right = band Power in specified freq of right trials
    %           bandpower_feat = band Power in specified freq of all 128
    %           trials - extraction of a feature for our Classifier

    if strcmpi(which_channel, "C3") == 1
        channel = 1;
    else
        channel = 2;
    end

    bandpower_left  = 10*log10(bandpower(left_data(:, :, channel)', fs, which_band));
    bandpower_right = 10*log10(bandpower(right_data(:,:, channel)', fs, which_band));

    bandpower_feat  = 10*log10(bandpower(data(:,:,channel)', fs, which_band));

end