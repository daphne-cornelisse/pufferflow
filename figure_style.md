This document provides guidelines for how you should create figures.

1. Use seaborn with sns.set_style("ticks").
2. Unless it is a heatmap or image, use sns.despine().
3. Add a transparent grid in the background .grid(True, alpha=0.3).
4. Use the default seaborn color palette for categorical data.
5. Unless specified otherwise, if I ask for 2 plots in a subplot, make it 2 cols, 1 row.
6. Use a dpi of 100 unless specified otherwise.