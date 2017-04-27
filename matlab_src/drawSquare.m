function drawSquare(x_corr, y_corr, window_size, color, label)
hold on
plot3([x_corr-window_size/2, x_corr+window_size/2], [y_corr-window_size/2, y_corr-window_size/2], [1 1], 'Color', color)
plot3([x_corr-window_size/2, x_corr+window_size/2], [y_corr+window_size/2, y_corr+window_size/2], [1 1], 'Color', color)
plot3([x_corr-window_size/2, x_corr-window_size/2], [y_corr-window_size/2, y_corr+window_size/2], [1 1], 'Color', color)
plot3([x_corr+window_size/2, x_corr+window_size/2], [y_corr-window_size/2, y_corr+window_size/2], [1 1], 'Color', color)
text(x_corr, y_corr, 1, label, 'FontSize', 24, 'Color', color)