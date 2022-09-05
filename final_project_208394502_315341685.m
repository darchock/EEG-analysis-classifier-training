% Final Project – Analysis of Motor Imagery Data
% submitted by; 208394502 & 315341685

%%
clear;          %clear all variables
close all;      %close all figures
clc;            %clear command window

%% setting parameters
freq              = 1:0.1:40; % [Hz]  
TITLE_FONT        = 22;
channel_vec       = {'C3' , 'C4'};
n_channel         = length(channel_vec); % num of Channels
c3                = 1;
c4                = 2;
left_row          = 3;
right_row         = 4;
sub_plots_for_vis = 20;

%% Part 1 - Visualization
%  Part 1.1 Loading the training data
load('motor_imagery_train_data.mat');

fs                = P_C_S.samplingfrequency;  % [Hz]
class_vec         = (P_C_S.attributename(left_row:right_row))'; % Left & Right Hands
n_class           = length(class_vec); % num of Classes
left_hand_trials  = P_C_S.attribute(left_row, :);
right_hand_trials = P_C_S.attribute(right_row, :);
training_data     = P_C_S.data(:, :, 1 : n_channel);        % extracting data only for C3 & C4
n_trials          = size(training_data, 1); % num of Trials
time_samples      = size(training_data, 2); % num of Samples
time_vec_sec      = (1:time_samples) / fs; % time vector in seconds

% Left Hand
left_trial       = find(left_hand_trials == 1); % finding the trials tagged - Left hand
left_hand_matrix = training_data(left_trial, :, :); % data matrix specifcly for left hand tagged trials
left_rand_trials = left_trial(randperm(sub_plots_for_vis));

% Right Hand
right_trial       = find(right_hand_trials == 1); % finding the trials tagged - Right hand
right_hand_matrix = training_data(right_trial, :, :); % data matrix specifcly for right hand tagged trials
right_rand_trials = right_trial(randperm(sub_plots_for_vis));

% part 1.2 - Visualizing the EEG signal in a single channel for trials from a single class
title_str = ['Visualizing EEG signal in a single channel'];
n_rows    = 5;
n_cols    = 4;

figure('Name', [title_str ' - left hand'], ...
       "Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
plot_EEG(left_hand_matrix, title_str, n_rows, n_cols, time_vec_sec, left_rand_trials, class_vec{1});

figure('Name', [title_str ' - right hand'], ...
       "Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
plot_EEG(right_hand_matrix, title_str, n_rows, n_cols, time_vec_sec, right_rand_trials, class_vec{2});

%% Part 2 - Power Spectra
% Difining parameters
title_power = ['Power spectrum using Welchs method for both hands'];
n_rows      = 2;
n_cols      = 1;
time_seg    = 3*fs : time_samples; % imagine time of trials
window      = 1*fs;                % length of window time
overlap     = 0.9 * window;        % size of overlap between windows

% Part 2.1 Calculating Power Spectrim from trials in each class
pwelch_left_c3  = pwelch(left_hand_matrix(:, time_seg, c3)', window, overlap, freq, fs);
pwelch_left_c4  = pwelch(left_hand_matrix(:, time_seg, c4)', window, overlap, freq, fs);
pwelch_right_c3 = pwelch(right_hand_matrix(:, time_seg, c3)', window, overlap, freq, fs);
pwelch_right_c4 = pwelch(right_hand_matrix(:, time_seg, c4)', window, overlap, freq, fs);
cell_left       = {pwelch_left_c3, pwelch_left_c4};
cell_right      = {pwelch_right_c3, pwelch_right_c4};

% Part 2.2 Plot a spectrum for each class and each channel
%          added a 1-standart_deviation confidence interval onto the plots
figure('Name', title_power, "Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
plot_power_spec(cell_left, cell_right, title_power, freq, time_vec_sec(time_seg(1)), ...
                time_vec_sec(time_seg(end)));

% Part 2.3 Spectograms
title_specto   = ['Spectrograms for each channel and hand'];
n_rows         = 2;
n_cols         = 2;
cell_spec      = cell(n_channel, n_class);
num_of_windows = floor((time_samples - window) / (window - overlap)); % num of windows the data will divide into

% Spectrograms of each class and each channel
figure('Name', title_specto, "Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
sgtitle(title_specto, "FontSize", TITLE_FONT, "FontWeight", "bold");
subplot_pos = 1;
for which_channel = 1 : n_channel
    [cell_spec{left_row-2, which_channel}, f, t] = compute_spec(left_hand_matrix, which_channel, ...
                                          window, floor(overlap), freq, fs, num_of_windows);
    plot_spectro(cell_spec{left_row-2, which_channel}, f, t, n_rows, n_cols, subplot_pos, "Left", ...
                 which_channel);
    subplot_pos = subplot_pos + 1;
    [cell_spec{right_row-2, which_channel}, f, t] = compute_spec(right_hand_matrix, which_channel, ...
                                          window, floor(overlap), freq, fs, num_of_windows);
    plot_spectro(cell_spec{right_row-2, which_channel}, f, t, n_rows, n_cols, subplot_pos, "Right", ...
                 which_channel);
    subplot_pos = subplot_pos + 1;
end

% Spectrogram of the mean difference between the 2 Channels; C3 & C4
%                    - were helpful during our features selection process
diff_cell = cell(n_channel, 1);
diff_str  = ['Difference between Left hand to Right hand Spectrograms in both Channels'];
figure("Name",diff_str,"Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
sgtitle(diff_str, "FontSize", TITLE_FONT, "FontWeight", "bold");
for channel = 1 : n_channel
    diff_cell{channel} = abs(cell_spec{1, channel} - cell_spec{2, channel});
    plot_spectro(diff_cell{channel}, f, t, n_channel, size(diff_cell,2), -1, ...
                 "Left hand to Right hand Spectrograms", channel);
end

%% Part 3 - Features - finding features for motor imagery
% Part 3.1 Choosing informative frequency bands
n_bands    = 3;
info_bands = cell(n_bands, 2);

% frequency [Hz]                % Which Electrode
info_bands{1,1} = [15 18.3];    info_bands{1,2} = "C3";
info_bands{2,1} = [15 17.2];    info_bands{2,2} = "C4";
info_bands{3,1} = [30.5 33];    info_bands{3,2} = "C4";

band_power_cell = cell(n_bands, n_bands);
for bp = 1 : n_bands
    [band_power_cell{bp,1}, band_power_cell{bp,2}, band_power_cell{bp,3}] = compute_band(training_data, ...
    left_hand_matrix(:, time_seg, :), right_hand_matrix(:, time_seg, :), fs, info_bands{bp,1}, info_bands{bp,2});
end

% Part 3.2 plot Histograms depicting the power distribution
title_histo = ['Band-Power feature histograms in 3 different frequency bands'];
figure('Name', title_histo, "Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
sgtitle(title_histo, "FontSize", TITLE_FONT, "FontWeight", "bold");
plot_histo(band_power_cell, info_bands, n_bands);

%% Part 3.2 adding more features besides power-bands

% Root Total Power - adding another feature
% for every class every Channel (C3 & C4) we'll compute the total root power, this
% std gives proportionnal assumption to the distance of the signal from avg
% we compute pwelch in windows from second 4 to 6
RTP_time    = 4* fs   : 6*fs; % Root Total Power computation time
RTP_window  = 0.5 * fs;       % length of RTP window
RTP_overlap = floor(49 / 50 * RTP_window); % size of overlap between windows in RTP computation
[RT_left_c3, RT_right_c3, RT_data_c3, RT_left_c4, RT_right_c4, RT_data_c4] = compute_rtp(training_data, ...
 left_hand_matrix, right_hand_matrix, RTP_time, RTP_window, RTP_overlap, freq, fs);

%% Spectral Entropy - adding another feature
se_start_time         = 4*fs; % Spectral Entropy start time - SE computation between second 4 to 6
[left_se_c3, time_se] = compute_se(left_hand_matrix, se_start_time, size(left_hand_matrix,1), fs, c3);
[right_se_c3, ~]      = compute_se(right_hand_matrix, se_start_time, size(right_hand_matrix,1), fs, c3);
[feat_se_c3, ~]       = compute_se(training_data, se_start_time, size(training_data,1), fs, c3);

%% STD in EEG Signal - adding another feature

% chosen time                          % channel
std_time1 = floor(4.25*fs : 4.9*fs);   %C4
std_time2 = 5*fs : 6*fs;               %C4   

[~, ~, data_sig_std1] = compute_std(training_data, left_hand_matrix, right_hand_matrix, ...
                       std_time1, c4);
[left_sig_std2, right_sig_std2, data_sig_std2] = compute_std(training_data, left_hand_matrix, right_hand_matrix, ...
                       std_time2, c4);

%% STD in Power Spectrum - adding another feature

% chosen frequencies            % chosen time                        % Channel
std_freq3 = [18: 0.05 : 21];    std_time3 = 2*fs : 3*fs;             % C4
std_freq4 = [30: 0.05 : 33.5];  std_time4 = floor(4.5*fs : 5.5*fs);  % C3
std_freq5 = [15: 0.05 : 17.3];  std_time5 = floor(4.5*fs : 6*fs);    % C4

% feature of STD in the Power Spectrum between the 2 hands in freq
% 18 : 21 [Hz] in second 2 till 2 in Channel C4
[std_feat_ps_data1, std_feat_ps_left1, std_feat_ps_right1] = compute_ps_std(training_data, ...
     left_hand_matrix, right_hand_matrix, std_time3, std_freq3, c4, window, overlap, fs);

% feature of std in power spectrum between the 2 hands in freq
% 30 : 33.5 [Hz] in second 4.5 till 5.5 in Channel C3
[std_feat_ps_data2, ~, ~] = compute_ps_std(training_data, left_hand_matrix, right_hand_matrix, std_time4, ...
                            std_freq4, c3, window, overlap, fs);

% feature of std in power spectrum between the 2 hands in freq
% 15 : 17.3 [Hz] in second 4.5 till 6 in Channel C4
[std_feat_ps_data3, std_feat_ps_left3, std_feat_ps_right3] = compute_ps_std(training_data, left_hand_matrix, ...
                    right_hand_matrix, std_time5, std_freq5, c4, window, overlap, fs);

%% Addintionnal BAND POWER feature

% chosen frequencies                % Channel
bp_freq     = [18.5 21]; % [Hz]     % C4
bp_left_c4  = 10*log10(bandpower(left_hand_matrix(:,:,c4)', fs, bp_freq));
bp_right_c4 = 10*log10(bandpower(right_hand_matrix(:,:,c4)', fs, bp_freq));
bp_data_c4  = 10*log10(bandpower(training_data(:,:,c4)', fs, bp_freq));

%% Part 3.3 Ploting Histograms of additional features chosen
YL     = '# Trials';
n_bins = 15;
figure("Name",'Additional Feature Histograms',"Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
sgtitle('Additional Feature Histograms', 'FontSize', TITLE_FONT, 'FontWeight', 'bold');
subplot(2,3,1);
plot_hist_feat(RT_left_c4, RT_right_c4, channel_vec{c4}, time_vec_sec(RTP_time(1)), ...
               time_vec_sec(RTP_time(end)), 'Root Total Power', 'square sum of PS from mean', YL, ...
               n_bins, 0.45);
subplot(2,3,2);
plot_hist_feat(left_se_c3, right_se_c3, channel_vec{c3}, time_vec_sec(se_start_time(1)), ...
               time_vec_sec(end), 'Spectral Entropy in EEG signal', 'Spectral Entropy', YL, ...
               n_bins, 0.006);
subplot(2,3,3);
plot_hist_feat(left_sig_std2, right_sig_std2, channel_vec{c4}, time_vec_sec(std_time2(1)), ...
               time_vec_sec(std_time2(end)), 'STD in EEG signal', 'std', YL, ...
               n_bins, 0.3);
subplot(2,3,4);
plot_hist_feat(std_feat_ps_left3, std_feat_ps_right3, channel_vec{c4}, time_vec_sec(std_time5(1)), ...
               time_vec_sec(std_time5(end)), 'STD in PS frequencies 15:17.3 [Hz]', 'std', YL, ...
               n_bins, 0.3);
subplot(2,3,5);
plot_hist_feat(std_feat_ps_left1, std_feat_ps_right1, channel_vec{c4}, time_vec_sec(std_time3(1)), ...
               time_vec_sec(std_time3(end)), 'STD in PS frequencies 18:21 [Hz]', 'std', YL, ...
               n_bins, 0.05);
subplot(2,3,6);
plot_hist_feat(bp_left_c4, bp_right_c4, channel_vec{c4}, 0, ...
               time_vec_sec(end), 'Band Power in frequencies 18.5:21 [Hz]', 'dB', YL, ...
               n_bins, 0.4);
legend('Left Hand', 'Right Hand', 'FontSize', 13, 'Position', [0.93 0.8 0.05 0.08]);


%% Part 3.4 features matrix including all features on all 128 trials % PCA
n_feat                 = 12; % num of features
features_matrix        = zeros(n_trials,n_feat);
features_matrix(:, 1)  = band_power_cell{1,3}(:);
features_matrix(:, 2)  = band_power_cell{2,3}(:);
features_matrix(:, 3)  = band_power_cell{3,3}(:);
features_matrix(:, 4)  = feat_se_c3(:);
features_matrix(:, 5)  = data_sig_std1(:);
features_matrix(:, 6)  = data_sig_std2(:);
features_matrix(:, 7)  = RT_data_c3(:);
features_matrix(:, 8)  = RT_data_c4(:);
features_matrix(:, 9)  = std_feat_ps_data1(:);
features_matrix(:, 10) = std_feat_ps_data2(:);
features_matrix(:, 11) = std_feat_ps_data3(:);
features_matrix(:, 12) = bp_data_c4(:);

features_matrix_norm   = zscore(features_matrix, 0, 1); % normalizing feature matrix :)

% PCA
title_pca = ['Principal Components Analysis on Normalized Features Classified by Hand'];
[coeff, score, ~, ~, explained] = pca(features_matrix_norm);
figure("Name",title_pca,"Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
sgtitle(title_pca, 'FontSize', TITLE_FONT, 'FontWeight', 'bold');
plot_pca(score(:,1:2), score(:,1:3), left_trial, right_trial);
disp(sum(explained(1:3))); % displaying percentage of variance explained in the 3 PC selected

%% Part 4 - Calssification
% Defining parmaters
n_selected_features    = 4;      % num of selected features (from the total 12 features)
labels_vec = zeros(n_trials, 1); % we'll tag left hand trials with 1 and right hand trials with 0
labels_vec(left_trial) = 1;
feat_names = {'Band Power 15 : 18.3 in C3', 'Band Power 15 : 17.2 in C4', 'Band Power 30.5 : 33 in C4', ...
     'Spectral Entropy fron second 4 to 6 in C3', 'std from second 4.25 to 4.9 in C4', ...
     'std from second 5 to 6 in C4', 'Root Total Power in C3', ...
     'Root Total Power in C4', 'std in PS freq 18 : 21 from second 2 to 3 in C4', ...
     'std in PS freq 30 : 33.5 from second 4.5 to 5.5 in C3', ...
     'std in PS freq 15 : 17.3 from second 4.5 to 6 in C4', 'Band Power 18.5 : 21 in C4'};
feat_channel = {channel_vec{c3}, channel_vec{c4}, channel_vec{c4}, channel_vec{c3}, channel_vec{c4}, ...
                channel_vec{c4}, channel_vec{c3}, channel_vec{c4}, channel_vec{c4}, channel_vec{c3}, ...
                channel_vec{c4}, channel_vec{c4}};
var_names = {'Feat_importance_ranking', 'Feat_channel', 'Feat_name'};

% Ranking our features & Selecting the best features in order to train our Classifier
[selected_features, selected_features_rank, features_rank] = select_features(features_matrix_norm, ...
                            n_selected_features, labels_vec, feat_names, feat_channel, var_names);

% Part 4.1 Training our Classifier
k               = 5;                   % chosen k for the k-fold croos-validation algorithm
decimal2percent = 100;                 % var for displaying reasons
k_fold_sz       = floor(n_trials / k); % set size of each k-fold group

% Randomizing trials
samples_rand             = randperm(n_trials);
labels_rand              = labels_vec(samples_rand);
feat_by_rank_rand_trials = selected_features(samples_rand, :);

% Let's train our Classifier & display it's results :)
train_my_classifier(feat_by_rank_rand_trials, labels_rand, k);


%% Part 4.3 Testing our Calssifier
testing_data     = load("motor_imagery_test_data.mat").data(:, :, 1 : n_channel);
n_samples        = size(testing_data, 1);
testing_features = zeros(n_samples, n_feat);

% Band Power Features
testing_band_cell = cell(1, n_bands);
for bp = 1 : n_bands
    [~, ~, testing_band_cell{bp,3}] = compute_band(testing_data, ...
    left_hand_matrix(:, time_seg, :), right_hand_matrix(:, time_seg, :), fs, info_bands{bp,1}, info_bands{bp,2});
end

% Spectral entropy Feature
[testing_se, ~]       = compute_se(testing_data, time_seg, size(testing_data,1), fs, c3);

% STD in EEG signal Feature
[~, ~, test_sig_std1] = compute_std(testing_data, left_hand_matrix, right_hand_matrix, ...
                       std_time1, c4);
[~, ~, test_sig_std2] = compute_std(testing_data, left_hand_matrix, right_hand_matrix, ...
                       std_time2, c4);

% Root Total Power Feature
[~, ~, RT_test_c3, ~, ~, RT_test_c4] = compute_rtp(testing_data, ...
 left_hand_matrix, right_hand_matrix, RTP_time, RTP_window, RTP_overlap, freq, fs);

% STD in Power Spectrum Feature
[test_ps_std1, ~, ~]  = compute_ps_std(testing_data, left_hand_matrix, right_hand_matrix, std_time3, ...
                            std_freq3, c4, window, overlap, fs);
[test_ps_std2, ~, ~]  = compute_ps_std(testing_data, left_hand_matrix, right_hand_matrix, std_time4, ...
                            std_freq4, c3, window, overlap, fs);
[test_ps_std3, ~, ~]  = compute_ps_std(testing_data, left_hand_matrix, right_hand_matrix, ...
                            std_time5, std_freq5, c4, window, overlap, fs);

% Band Power Feature
test_bp  = 10*log10(bandpower(testing_data(:,:,c4)', fs, bp_freq));

testing_features(:, 1)  = testing_band_cell{1, 3}(:);
testing_features(:, 2)  = testing_band_cell{2, 3}(:);
testing_features(:, 3)  = testing_band_cell{3, 3}(:);
testing_features(:, 4)  = testing_se(:);
testing_features(:, 5)  = test_sig_std1(:);
testing_features(:, 6)  = test_sig_std2(:);
testing_features(:, 7)  = RT_test_c3(:);
testing_features(:, 8)  = RT_test_c4(:);
testing_features(:, 9)  = test_ps_std1(:);
testing_features(:, 10) = test_ps_std2(:);
testing_features(:, 11) = test_ps_std3(:);
testing_features(:, 12) = test_bp(:);

testing_features_norm     = zscore(testing_features, 0, 1);
testing_features_selected = testing_features_norm(:, selected_features_rank);
testing_data_classify     = classify(testing_features_selected, selected_features, labels_vec, 'linear');

% Displaying parameters;
fprintf(['Num of Channels -\t    ' num2str(n_channel) ...
         '\nNum of training samples -\t   ' num2str(size(training_data,1)) ...
         '\nNum of testing samples -\t    ' num2str(size(testing_data,1)) ...
         '\nSampling rate -\t     ' num2str(fs) ...
         '\nWindow size -\t       ' num2str(window) ...
         '\nOverlap size -\t      ' num2str(overlap) ...
         '\nDisplayed Frequencies :\t     ' num2str(freq(1)) ' - ' num2str(freq(end)) ...
         '\nK fold -\t  ' num2str(k) ...
         '\n\n']);

%% Part 4.2 Improving Accuracy
% why we chose set of 4 features for training our classifier
for test = 1 : 4
    n_selected_features = n_selected_features + 2;
    feat_test           = features_matrix_norm(:, features_rank(1:n_selected_features));

    samples_rand             = randperm(n_trials);
    labels_rand              = labels_vec(samples_rand);
    feat_by_rank_rand_trials = feat_test(samples_rand, :);

    train_my_classifier(feat_by_rank_rand_trials, labels_rand, k);
end

%% Apendix Plots
% Histograms of Root Total Power, why we chose it's own window and overlap
% val -- shows better sicrimination:
[RT_left_c3_reg, RT_right_c3_reg, ~, RT_left_c4_reg, RT_right_c4_reg, ~] = compute_rtp(training_data, ...
 left_hand_matrix, right_hand_matrix, RTP_time, window, overlap, freq, fs);

XL = 'square sum of PS from mean';
appendix_str = 'Root Total Power with 1 sec window and 0.9 overlap size';
appendix_str_C3 = [appendix_str ' in Channel ' channel_vec{1}];
figure("Name", appendix_str_C3, "Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
histogram(RT_left_c3_reg, n_bins, "BinWidth", 0.35);
hold on;
histogram(RT_right_c3_reg, n_bins, "BinWidth", 0.35);
title(appendix_str_C3, "FontSize", TITLE_FONT, "FontWeight", "bold");
xlabel(XL, "FontSize", 18);
ylabel(YL, "FontSize", 18);
hold off;

appendix_str_C4 = [appendix_str ' in Channel ' channel_vec{2}];
figure("Name", appendix_str_C4, "Color", 'w', 'Units', 'normalized', 'windowstate','maximized');
histogram(RT_left_c4_reg, n_bins, "BinWidth", 0.35);
hold on;
histogram(RT_right_c4_reg, n_bins, "BinWidth", 0.35);
title(appendix_str_C4, "FontSize", TITLE_FONT, "FontWeight", "bold");
xlabel(XL, "FontSize", 18);
ylabel(YL, "FontSize", 18);
hold off;

% end