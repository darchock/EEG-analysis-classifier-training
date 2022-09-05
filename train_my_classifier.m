function [] = train_my_classifier(feat_by_rank_rand_trials, labels_rand, k);
    % func purpose - train our Classifier with k-fold cross-validation algo
    % @ input: feat_by_rank_rand_trials = selected_features matrix (4 features total)
    %                                     already ranked with his n_trials randomized.
    %          labels_rand              = tagged trials randomized, 1 is Left 0 Right
    %          k                        = k'th value chosen before
    % @ output: displays training & validation accuracy mean & std of our
    %           Classifier
    
    % parameters
    counter                         = 1;
    [n_trials, n_selected_features] = size(feat_by_rank_rand_trials);
    decimal2percent                 = 100;
    k_fold_sz                       = floor(n_trials / k);

    % data allocations
    which_tagged_class              = zeros(k_fold_sz, k);
    error_per                       = zeros(1, k);
    validation_acc                  = zeros(1 , k);

    for which_fold = 1 : k
        fold_data_idx = (counter : k_fold_sz*which_fold);
        train_idx = setdiff(1:n_trials, fold_data_idx);
        [which_tagged_class(:, which_fold) , error_per(which_fold)]  =  ...
          classify(feat_by_rank_rand_trials(fold_data_idx, :) ,feat_by_rank_rand_trials(train_idx, :), ...
          labels_rand(train_idx) , 'linear');
        counter = counter + k_fold_sz;
        validation_acc(which_fold) = ...
                      sum(which_tagged_class(:, which_fold) == labels_rand(fold_data_idx)) / k_fold_sz;
    end

    % Train accuracy computation:
    training_acc = 1 - error_per;
    training_mean = mean(training_acc);
    training_std = std(training_acc);

    % Validation accuracy computation:
    validation_mean = mean(validation_acc);
    validation_std = std(validation_acc);

    % Displaying the results:
    fprintf("Features Set Size: %d \n", n_selected_features);
    fprintf("Validation Accuracy: %.3f±%.3f%% \n", validation_mean * decimal2percent ...
                                                 , validation_std * decimal2percent);
    fprintf("Train Accuracy: %.3f±%.3f%% \n", training_mean * decimal2percent, ...
                                              training_std * decimal2percent);
end