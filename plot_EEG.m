function [] = plot_EEG(training_matrix, title_str, n_rows, n_cols, time_vec, ...
                        which_trials, class)
    % func purpose - plotting visulization of 20 trials EEG signal as a function over time
    % @ input: training_matrix = left_hand_matrix or right_hand_matrix. 
    %          title_str = title string for the plot. n_row & n_cols = figure format of subplot.
    %          time_vec = samples vector time in seconds.
    %          which_class = left_trials or right_trials.
    %          class = left or right hand
    % @ output: plots Graph

    legend_pos      = [0.9 0.9 0.05 0.08];
    ylimit          = [-20,20];
    sub_title_font  = 12;
    title_font      = 18;
    axes_font       = 16;
    
    for tr = 1 : length(which_trials) 
        which_trial = which_trials(tr);
        trial_c3    = training_matrix(tr, :, 1);
        trial_c4    = training_matrix(tr, :, 2);

        subplot(n_rows, n_cols, tr);
        hold on;
        plot(time_vec, trial_c3, 'Color', 'b' ,"LineWidth", 0.01);
        plot(time_vec, trial_c4, 'Color', 'r' ,"LineWidth", 0.01);
        ylim(ylimit);
        hold off;
        subtitle(['Trial #' num2str(which_trial)], "FontSize", sub_title_font, "FontWeight", "bold");
        if mod(tr, n_cols) == 1
            ylabel('[mV]', 'FontSize', axes_font, 'FontWeight', 'bold');
        end
        if tr > 16
            xlabel('Time [sec]', 'FontSize', axes_font, 'FontWeight', 'bold');
        end
        if tr == length(which_trials) 
            legend('C3', 'C4', "Position", legend_pos, "FontSize", axes_font);
            sgtitle([title_str ' - ' class ' hand'], 'FontSize', title_font, 'FontWeight', 'bold');
        end

    end

end