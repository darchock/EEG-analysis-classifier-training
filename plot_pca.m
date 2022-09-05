function [] = plot_pca(data_2D, data_3D, left_trials, right_trials)
    % func purpose - plots the Principle Component Graph, 1 in 2D 1 in 3D
    % @ input: data_2D = data of the first two principle components from
    %                    the pca() func
    %          data_3D = data of the first three principle components from
    %                    the pca() func
    %          left_trials = trials indices of tagged left hand trials only
    %          right_trials = trials indices of tagged right hand trials only
    % @ output: plots Graph displaying the PC in 2D and 3D

    font_title  = 18;
    font_axes   = 14;
    left_color  = '#A2142F';
    right_color = '#4DBEEE';
    legend_pos  = [0.9 0.8 0.05 0.08];
    sz          = 35; % size of markers

    % plotting the 2D pca graph - plotting the 2 'best' components
    subplot(2,1,1);
    s2_1 = scatter(data_2D(left_trials, 1), data_2D(left_trials, 2));
    s2_1.ColorVariable = left_color;
    s2_1.SizeData = sz;
    hold on;
    s2_2 = scatter(data_2D(right_trials, 1), data_2D(right_trials, 2));
    s2_2.ColorVariable = right_color;
    s2_2.SizeData = sz;
    xlabel('PC_{1}', 'FontSize', font_axes);
    ylabel('PC_{2}', 'FontSize', font_axes);
    title('2D PCA of best 2 features', "FontSize", font_title, "FontWeight", "bold", "FontAngle", "italic");
    hold off;
    % plotting the 3D pca graph - plotting the 3 'best' components
    subplot(2,1,2);
    s3_1 = scatter3(data_3D(left_trials, 1), data_3D(left_trials, 2), data_3D(left_trials, 3));
    s3_1.ColorVariable = left_color;
    s3_1.SizeData = sz;
    hold on;
    grid on;
    s3_2 = scatter3(data_3D(right_trials, 1), data_3D(right_trials, 2), data_3D(right_trials, 3));
    s3_2.ColorVariable = right_color;
    s3_2.SizeData = sz;
    xlabel('PC_{1}', 'FontSize', font_axes);
    ylabel('PC_{2}', 'FontSize', font_axes);
    zlabel('PC_{3}', 'FontSize', font_axes);
    title('3D PCA of best 3 features', "FontSize", font_title, "FontWeight", "bold", "FontAngle", "italic");
    hold off;
    legend('Left Hand', 'Right Hand', 'FontSize', font_axes, "Position", legend_pos);
    
end