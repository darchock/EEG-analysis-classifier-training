function [] = plot_hist_feat(l_hist, r_hist, which_channel, start, ending, which_feat, XL, YL, n_bins, ...
                             bin_width)
    % func purpose - plots Features Histograms Graph in order to see
    %                diffrentitaion between left and right trials
    % @ input: l_hist = first data to be plotted on Histogram
    %          r_hist = second data to be plotted on Histogram
    %          which_channel = from which channel the data was extracted
    %          start = start of time segment from which we extracted the data
    %          ending = end of time segment from which we extracted the data
    %          which_feat = string describing the specific feature to be displayed
    %          XL = string to be displayed on xlabel
    %          YL = string to be displayed on ylabel
    %          n_bins = num of bins the data is divided
    %          bin_width = width of the bins displayed
    % @ ouput: plots Features Histograms Graph
    
    title_str   = {['Feature ' which_feat], [' from sec ' num2str(start) ':' num2str(ending) ...
                    ' in ' which_channel]};
    left_color  = '#A2142F';
    right_color = '#4DBEEE';
    font_S      = 16;
    font_title  = 15;

    histogram(l_hist, n_bins, 'FaceColor', left_color, 'BinWidth', bin_width);
    hold on;
    histogram(r_hist, n_bins, 'FaceColor', right_color, 'BinWidth', bin_width);
    xlabel(XL, 'FontSize', font_S);
    ylabel(YL, "FontSize", font_S);
    title(title_str, 'FontSize', font_title, 'FontWeight', 'bold', 'FontAngle', 'italic');
    hold off;

end