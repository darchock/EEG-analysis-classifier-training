function [] = plot_power_spec(cell_left, cell_right, title_power, freq, start_t, end_t)
    % func purpose - plot Power Spectrum Graph who discriminates between C3
    % & C4 Power Spectrum in Left & Right Hands.
    % @ input: cell_left = cellArray of pwelch for both channels - left hand
    %          cell_right = cellArray of pwelch for both channels - right hand
    %          title_power = string for title of the Graph
    %          freq = frequency spectrum in [Hz]
    %          start_t = start time in [sec]
    %          end_t = ending time in [sec]
    % @ output: plots Graph

    
        n_rows        = 2;
        n_cols        = 1;
        font_title    = 20;
        font_subtitle = 17;
        font_axes     = 15;
        line          = 2;
        shade         = 0.4;

        for channel = 1 : length(cell_left) 

            subplot(n_rows, n_cols, channel);
            hold on;
            % converting to dB
            spec_left       = 10*log10(cell_left{channel});
            spec_left_mean  = mean(spec_left, 2);
            spec_left_std   = std(spec_left, 0, 2);
            spec_right      = 10*log10(cell_right{channel});
            spec_right_mean = mean(spec_right, 2);
            spec_right_std  = std(spec_right, 0, 2);
            polyg_std_left  = polyshape([freq, flip(freq)], ...
                             [spec_left_mean + spec_left_std; flip(spec_left_mean - spec_left_std)]');
            polyg_std_right = polyshape([freq, flip(freq)], ...
                             [spec_right_mean + spec_right_std; flip(spec_right_mean - spec_right_std)]');

            % adding std to the plot using fixed matlab func polyshape
            plot(freq, spec_left_mean,"Color", '#0072BD', "LineWidth", line, "LineStyle", "-");
            plot(polyg_std_left, "FaceColor", '#4DBEEE', "EdgeColor", 'none', "FaceAlpha", shade);
            plot(freq, spec_right_mean, "Color", '#D95319', "LineWidth", line, "LineStyle", "--");
            plot(polyg_std_right, "FaceColor", '#EDB120', "EdgeColor", 'none', "FaceAlpha", shade);
            hold off;
            
            subtitle(['Power spectra for both hands in channel C' num2str(channel+2) ' from second ' ...
                      num2str(start_t) ' till ' num2str(end_t)], "FontSize", font_subtitle, "FontAngle", ...
                      "italic");
            ylabel('Power [dB]', 'FontSize', font_axes);
            legend('left hand', 'left std', 'right hand', 'right std', "FontSize", font_axes, ...
                   "Location", "best");
            sgtitle(title_power, "FontSize", font_title, "FontWeight", "bold");
            if channel == length(cell_left)
                xlabel('Frequency [Hz]', 'FontSize', font_axes); 
            end
        end

end