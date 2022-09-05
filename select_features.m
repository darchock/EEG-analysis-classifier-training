function [selected_features, selected_features_rank, features_rank] = select_features(features_normalizied, ...
          n_selected_features, labels, feat_names, feat_elec, var_names)
    % func purpose - 1, Ranking Features extracted from the data by weights
    % 2, select the set size chosen before of features by ranking
    % 3, save as a csv file the features ranking table
    % @ input: features_normalizied = features matrix after zscore (norm)
    %          n_selected_features = features set size chosen before
    %          labels = bonary labels vector, 1 is Left 0 is Right hand
    %          feat_names = vector of features name
    %          feat_elec = vector of channel of which each features extracted
    %          var_names = csv table col names
    % @ ouput: selected_features = only the 4 best features (norm)
    %          selected_features_rank = the ranking of the 4 best features
    %          features_rank = the ranking of all 12 features
        
    % performing fscnca
    feat_importance        = fscnca(features_normalizied, labels);            % performing the NCA algorithm
    [~, features_rank]     = sort(feat_importance.FeatureWeights, 'descend'); % sorting Ranking output from the NCA algorithm
    selected_features_rank = features_rank(1 : n_selected_features);          % now is ranking of all 4 best features
    selected_features      = features_normalizied(:, selected_features_rank); % only the 4th best features (norm)

    [~, linear_independent_cols] = rref(selected_features);                   % returns the reduced row echelon form using Gauss-Jordan elimination
    selected_features = selected_features(:, linear_independent_cols);

    % saving csv file of table sorted by features ranking to display later
    % in report
    Feat_names_table  = feat_names(:, features_rank);
    Feat_channel_table = feat_elec(:, features_rank);
    Feat_ranking_table = table((1:length(features_rank))', Feat_channel_table', Feat_names_table', ...
                         'VariableNames', var_names);
    writetable(Feat_ranking_table, 'features_ranking_details.csv');

    % checking the indpendent cols are all of the cols which means there
    % were no dependent cols at all
    if length(linear_independent_cols) ~= length(selected_features_rank)
        selected_features_rank = selected_features_rank(linear_independent_cols);
    end

end