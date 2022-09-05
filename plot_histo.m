 function [] = plot_histo(band_power, band_info, n_bands)
    % func purpose - plots the Histograms of the 3 frequency Bands computed
    % and depicted the differentiation between Left and Right hand in them.
    % @ input: band_power = cell Array of Band Power computed for each band
    %          band_info = info on each band; range of freq, from which
    %          electrode was extracted, n_bands = num of bands.
    % @ output: plots Histogram's Graph of the 3 chosen bands
    
    col         = 1;
    left_color  = '#A2142F';
    right_color = '#4DBEEE';
    n_bins      = 15;
    font_title  = 18;
    font_axes   = 17;
    left        = 1;
    right       = 2;
    channel_col = 2;
    bin_width   = 0.35;
    legend_pos  = [0.9 0.9 0.05 0.08];

    for hist = 1 : n_bands
        subplot(n_bands, col, hist);
        
        histogram(band_power{hist, left}, n_bins, 'FaceColor', left_color, 'BinWidth', bin_width);
        hold on;
        histogram(band_power{hist, right}, n_bins, 'FaceColor', right_color, 'BinWidth', bin_width);
        hold off;

        which_channel = band_info{hist, channel_col};
        title_str = 'Frequency band in ' + which_channel + ' from ' + num2str(band_info{hist, col}(1)) + ...
                    ' to ' + num2str(band_info{hist, col}(2)) + ' [Hz]'; 
        title(title_str, 'FontSize', font_title, 'FontWeight', 'bold', 'FontAngle', 'italic');
        ylabel('# Trials', 'FontSize', font_axes);

        if hist == n_bands
            xlabel('dB', 'FontSize', font_axes);
            legend('Left hand', 'Right hand', 'FontSize', font_axes, 'Position', legend_pos);
        end
    end

end