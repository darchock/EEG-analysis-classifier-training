function [] = plot_spectro(spec, f, T, n_rows, n_cols, sub_pos, hand, elec)
    % func purpose - plot Spectrogram Graph using the outputs of
    % 'compute_spec'
    % @ input: spec = mean of spectrogram() in [dB] output of 'compute_spec'
    %          f = frequency spectrum
    %          T = vector of time instants
    %          n_rows & n_cols = format of the subplots on figure displayed
    %          sub_pos = indicator on which spot on figure to plot, also
    %                    indicator of disintiguishing between differences & splitted
    %                    Spectrograms plots
    %          hand = string for title
    %          elec = which electrode in depicted C3 or C4
    % @ output: plots Spectrogram Graphs - differences & splitted 
    
    font_title = 17;
    font_axes  = 15;
    font_cb    = 13;
    flag       = 0;
    if elec == 1
        elec_str = string('C3');
    else
        elec_str = string('C4');
    end

    if sub_pos == -1
        title_str = "Difference between " + hand + ' in Channel ' + elec_str;
        sub_pos   = elec;
        flag      = 1;
    else
        title_str = "Spectrogram of " + hand + ' hand in channel - ' + elec_str;
    end

    subplot(n_rows, n_cols, sub_pos);
    imagesc(T, f, spec);
    axis normal;
    set(gca, 'YDir', 'normal', 'FontSize', font_axes);
    hold on;
    colormap jet;
    title(title_str, "FontSize", font_title, "FontAngle", "italic", "FontWeight", "normal");

    % adding labels according to position on figure
    if flag == 0
        if mod(sub_pos,2) == 1
            ylabel('Frequency [Hz]', "FontSize", font_axes);
        elseif mod(sub_pos,2) == 0
            cb          = colorbar;
            cb.Location = "eastoutside";
            cb.FontSize = font_cb;
            ylabel(cb, 'Power [dB]', 'FontSize', font_cb);
        end

        if sub_pos >= 3
            xlabel("Time [Sec]", "FontSize", font_axes);
        end
    else
        ylabel('Frequency [Hz]', "FontSize", font_axes);
        cb = colorbar;
        cb.Location = "eastoutside";
        cb.FontSize = font_cb;
        ylabel(cb, 'Power [dB]', 'FontSize', font_cb);
        if elec == 2
            xlabel("Time [Sec]", "FontSize", font_axes);
            cb          = colorbar;
            cb.Location = "eastoutside";
            cb.FontSize = font_cb;
            ylabel(cb, 'Power [dB]', 'FontSize', font_cb);
        end
    end
    
end