# -*- coding: utf-8 -*-
"""
Visualization Module for Sphingolipid Analysis
===============================================

Generates publication-quality figures including:
1. Bar plots with error bars (group comparisons)
2. Stacked bar plots (composition)
3. Heatmaps (correlation, group patterns)
4. Box/violin plots
5. Pie charts (percentage breakdown)
6. Ratio comparison plots
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palettes for sphingolipid classes
SPHINGO_CLASS_COLORS = {
    'ceramide': '#1f77b4',
    'dihydroceramide': '#aec7e8',
    'sphingomyelin': '#2ca02c',
    'sphingoid_base': '#d62728',
    'sphingoid_base_phosphate': '#ff9896',
    'hexosylceramide': '#9467bd',
    'ceramide_1_phosphate': '#8c564b',
}

CHAIN_LENGTH_COLORS = {
    'short': '#1f77b4',
    'medium': '#2ca02c',
    'long': '#ff7f0e',
    'very_long': '#d62728',
}

# Default palette - can be overridden
DEFAULT_PALETTE = "Set2"


def get_group_palette(palette_name: str = "Set2", n_colors: int = 10):
    """Get a color palette by name."""
    import seaborn as sns
    return sns.color_palette(palette_name, n_colors)


class SphingolipidVisualizer:
    """Generate visualizations for sphingolipid data analysis."""
    
    def __init__(
        self,
        figsize_single: Tuple[float, float] = (8, 6),
        figsize_wide: Tuple[float, float] = (12, 6),
        figsize_tall: Tuple[float, float] = (8, 10),
        color_palette: str = "Set2",
        style: str = "whitegrid"
    ):
        self.figsize_single = figsize_single
        self.figsize_wide = figsize_wide
        self.figsize_tall = figsize_tall
        self.palette_name = color_palette
        self.group_palette = get_group_palette(color_palette, 10)
        sns.set_style(style)
        sns.set_palette(color_palette)

        # Fixed species->color map derived from the full canonical panel sorted
        # alphabetically, so each species gets the same color in every experiment
        # regardless of which subset is detected or how percentages rank them.
        # Hues step by the golden ratio so alphabetically-adjacent species land
        # far apart on the color wheel (prevents two neighbours in a sorted pie
        # from looking similar). Saturation/value are dimmed below the default
        # hsv colormap so slices don't look neon.
        import colorsys
        from config.sphingolipid_species import get_all_species
        canonical_species = sorted(get_all_species())
        golden_ratio_conjugate = 0.61803398875
        saturation = 0.60
        value = 0.85
        self.species_color_map: Dict[str, tuple] = {}
        for i, sp in enumerate(canonical_species):
            hue = (i * golden_ratio_conjugate) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            self.species_color_map[sp] = (r, g, b, 1.0)
        self.species_color_map['Other'] = '#F5F0E8'
    
    def plot_group_comparison(
        self,
        data: pd.DataFrame,
        value_col: str,
        group_col: str,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        plot_type: str = "bar",
        show_points: bool = True,
        significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        colors: Optional[Dict[str, str]] = None,
        ax: Optional[plt.Axes] = None,
        log_scale: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create group comparison plot."""
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        if figsize is None:
            figsize = self.figsize_single
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Get valid groups only
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        palette = colors if colors else dict(zip(groups, self.group_palette[:len(groups)]))
        
        # Prepare plot data - apply log10 transform if requested
        plot_col = value_col
        plot_ylabel = ylabel or value_col
        
        if log_scale:
            log_col = f'{value_col}_log10'
            min_positive = data[value_col][data[value_col] > 0].min()
            floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
            data[log_col] = np.log10(data[value_col].clip(lower=floor_val))
            plot_col = log_col
            plot_ylabel = f'log₁₀({ylabel or value_col})'
        
        if plot_type == "bar":
            means = data.groupby(group_col)[plot_col].mean()
            sems = data.groupby(group_col)[plot_col].sem()
            
            x = range(len(groups))
            bars = ax.bar(x, [means[g] for g in groups], 
                         yerr=[sems[g] for g in groups],
                         capsize=5, color=[palette.get(g, '#888888') for g in groups],
                         edgecolor='black', linewidth=0.5, alpha=0.8)
            
            if show_points:
                for i, g in enumerate(groups):
                    group_data = data[data[group_col] == g][plot_col]
                    jitter = np.random.normal(0, 0.05, len(group_data))
                    ax.scatter(i + jitter, group_data, color='black', 
                              alpha=0.5, s=20, zorder=3)
            
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            
        elif plot_type == "box":
            sns.boxplot(data=data, x=group_col, y=plot_col, ax=ax,
                       hue=group_col, palette=palette, order=groups, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                            color='black', alpha=0.5, size=4, order=groups)
        elif plot_type == "strip":
            sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                         hue=group_col, palette=palette, order=groups,
                         size=6, alpha=0.7, legend=False)
            # Add mean ± SEM lines
            means = data.groupby(group_col)[plot_col].mean()
            sems = data.groupby(group_col)[plot_col].sem()
            for i, g in enumerate(groups):
                if g in means.index:
                    ax.hlines(means[g], i - 0.25, i + 0.25, color='black', linewidth=1.5, zorder=5)
                    ax.errorbar(i, means[g], yerr=sems[g], color='black',
                               capsize=5, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)

        elif plot_type == "violin":
            sns.violinplot(data=data, x=group_col, y=plot_col, ax=ax,
                          hue=group_col, palette=palette, order=groups, 
                          inner='box', legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                            color='black', alpha=0.5, size=4, order=groups)
        
        if significance_pairs:
            self._add_significance_bars(ax, groups, significance_pairs, data, plot_col, group_col)
        
        ax.set_xlabel('')
        ax.set_ylabel(plot_ylabel)
        ax.set_title(title or f'{value_col} by {group_col}')
        
        return fig, ax
    
    def _add_significance_bars(self, ax, groups, pairs, data, value_col, group_col):
        """Add significance bars to plot."""
        y_max = data[value_col].max()
        y_range = data[value_col].max() - data[value_col].min()
        bar_height = y_range * 0.05
        
        group_idx = {g: i for i, g in enumerate(groups)}
        
        for i, (g1, g2, annotation) in enumerate(pairs):
            if g1 not in group_idx or g2 not in group_idx:
                continue
            
            x1, x2 = group_idx[g1], group_idx[g2]
            y = y_max + (i + 1) * bar_height * 2
            
            ax.plot([x1, x1, x2, x2], [y - bar_height/2, y, y, y - bar_height/2],
                   color='black', linewidth=1)
            ax.text((x1 + x2) / 2, y + bar_height/4, annotation,
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(top=y_max + (len(pairs) + 1) * bar_height * 2.5)
    
    def plot_composition_stacked(
        self,
        data: pd.DataFrame,
        group_col: str,
        sphingolipid_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        as_percentage: bool = True,
        legend_outside: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create stacked bar plot showing sphingolipid composition."""
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        if figsize is None:
            figsize = self.figsize_wide
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        available_cols = [c for c in sphingolipid_cols if c in data.columns]
        
        if not available_cols:
            ax.text(0.5, 0.5, 'No sphingolipid data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig, ax
        
        group_means = data.groupby(group_col)[available_cols].mean()
        
        if as_percentage:
            row_sums = group_means.sum(axis=1)
            # Avoid division by zero
            row_sums = row_sums.replace(0, np.nan)
            group_means = group_means.div(row_sums, axis=0) * 100
            group_means = group_means.fillna(0)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(available_cols)))
        group_means.plot(kind='bar', stacked=True, ax=ax, color=colors,
                        edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)' if as_percentage else 'Concentration')
        ax.set_title(title or 'Sphingolipid Composition by Group')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        if legend_outside:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                     borderaxespad=0, fontsize=8)
        else:
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        method: str = 'spearman',
        figsize: Optional[Tuple[float, float]] = None,
        annot: bool = True,
        mask_upper: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create correlation heatmap."""
        if figsize is None:
            figsize = self.figsize_single
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        corr = data[columns].corr(method=method)
        
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                   annot=annot, fmt='.2f', square=True, ax=ax,
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'size': 8} if annot else {})
        
        ax.set_title(title or f'{method.capitalize()} Correlation Matrix')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_ratio_comparison(
        self,
        data: pd.DataFrame,
        ratio_col: str,
        group_col: str,
        title: Optional[str] = None,
        log_scale: bool = False,
        reference_line: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: str = "violin",
        show_points: bool = True,
        significance_pairs: Optional[List[Tuple[str, str, str]]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create comparison plot for a sphingolipid ratio.
        
        Args:
            data: DataFrame with ratio and group columns
            ratio_col: Column name for the ratio values
            group_col: Column name for grouping
            title: Plot title
            log_scale: Whether to log10 transform the data
            reference_line: Y value for reference line (default None)
            figsize: Figure size
            plot_type: 'violin', 'box', 'bar', or 'strip'
            show_points: Whether to show individual data points
            significance_pairs: List of (group1, group2, annotation) for significance brackets
        """
        if figsize is None:
            figsize = self.figsize_single
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        colors = dict(zip(groups, self.group_palette[:len(groups)]))
        
        # Prepare plot data - apply log10 transform if requested
        plot_col = ratio_col
        ylabel = ratio_col
        
        if log_scale:
            # Create log10 transformed column, handling zeros
            log_col = f'{ratio_col}_log10'
            min_positive = data[ratio_col][data[ratio_col] > 0].min()
            floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
            data[log_col] = np.log10(data[ratio_col].clip(lower=floor_val))
            plot_col = log_col
            ylabel = f'log₁₀({ratio_col})'
        
        # Plot based on type
        if plot_type == "violin":
            sns.violinplot(data=data, x=group_col, y=plot_col, ax=ax,
                          hue=group_col, palette=colors, order=groups, 
                          inner=None, alpha=0.3, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=colors, order=groups, 
                             size=6, alpha=0.7, legend=False)
            # Add mean lines
            means = data.groupby(group_col)[plot_col].mean()
            for i, g in enumerate(groups):
                if g in means.index:
                    ax.hlines(means[g], i-0.3, i+0.3, color='black', linewidth=2)
                    
        elif plot_type == "box":
            sns.boxplot(data=data, x=group_col, y=plot_col, ax=ax,
                       hue=group_col, palette=colors, order=groups, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=colors, order=groups,
                             size=4, alpha=0.6, legend=False)

        elif plot_type == "strip":
            sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                         hue=group_col, palette=colors, order=groups,
                         size=6, alpha=0.7, legend=False)
            # Add mean ± SEM lines
            means = data.groupby(group_col)[plot_col].mean()
            sems = data.groupby(group_col)[plot_col].sem()
            for i, g in enumerate(groups):
                if g in means.index:
                    ax.hlines(means[g], i - 0.25, i + 0.25, color='black', linewidth=1.5, zorder=5)
                    ax.errorbar(i, means[g], yerr=sems[g], color='black',
                               capsize=5, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)

        elif plot_type == "bar":
            sns.barplot(data=data, x=group_col, y=plot_col, ax=ax,
                       hue=group_col, palette=colors, order=groups, 
                       capsize=0.1, legend=False)
            if show_points:
                sns.stripplot(data=data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=colors, order=groups,
                             size=4, alpha=0.6, legend=False)
        
        # Add reference line (only makes sense if not log transformed)
        if reference_line is not None and not log_scale:
            ax.axhline(y=reference_line, color='gray', linestyle='--', 
                      linewidth=1, label=f'Reference ({reference_line})')
            ax.legend()
        
        # Add significance brackets if provided
        if significance_pairs:
            self._add_significance_bars(ax, groups, significance_pairs, data, plot_col, group_col)
        
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        ax.set_title(title or f'{ratio_col} by {group_col}')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_multi_panel_groups(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        group_col: str,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: str = "box",
        sharey: bool = False
    ) -> plt.Figure:
        """Create multi-panel figure comparing groups across multiple variables."""
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))
        
        if figsize is None:
            figsize = (ncols * 4, nrows * 3.5)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Get valid groups only (no NaN)
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        palette = dict(zip(groups, self.group_palette[:len(groups)]))
        
        for i, col in enumerate(value_cols):
            if col not in data.columns:
                continue
            
            if plot_type == "box":
                sns.boxplot(data=data, x=group_col, y=col, ax=axes[i],
                          hue=group_col, palette=palette, order=groups, legend=False)
            elif plot_type == "violin":
                sns.violinplot(data=data, x=group_col, y=col, ax=axes[i],
                             hue=group_col, palette=palette, order=groups, 
                             inner='box', legend=False)
            elif plot_type == "strip":
                sns.stripplot(data=data, x=group_col, y=col, ax=axes[i],
                             hue=group_col, palette=palette, order=groups,
                             size=6, alpha=0.7, legend=False)
                # Add mean ± SEM lines
                grp_means = data.groupby(group_col)[col].mean()
                grp_sems = data.groupby(group_col)[col].sem()
                for j, g in enumerate(groups):
                    if g in grp_means.index:
                        axes[i].hlines(grp_means[g], j - 0.25, j + 0.25, color='black', linewidth=1.5, zorder=5)
                        axes[i].errorbar(j, grp_means[g], yerr=grp_sems[g], color='black',
                                        capsize=5, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)
            elif plot_type == "bar":
                means = data.groupby(group_col)[col].mean()
                sems = data.groupby(group_col)[col].sem()
                x = range(len(groups))
                axes[i].bar(x, [means[g] for g in groups],
                          yerr=[sems[g] for g in groups],
                          capsize=3, color=[palette[g] for g in groups],
                          edgecolor='black', linewidth=0.5)
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(groups)
            
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('')
            if i % ncols == 0:
                axes[i].set_ylabel('Concentration')
            else:
                axes[i].set_ylabel('')
            axes[i].tick_params(axis='x', rotation=45)
        
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_multi_panel_groups_with_stats(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        group_col: str,
        stats_results: Dict,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: str = "box",
        sharey: bool = False,
        max_sig_bars: int = 3,
        log_scale: bool = False,
        show_points: bool = True,
        ylabel: str = "Concentration"
    ) -> plt.Figure:
        """
        Create multi-panel figure with significance annotations.
        
        Args:
            data: DataFrame with data
            value_cols: Columns to plot
            group_col: Grouping column
            stats_results: Dict of statistical results keyed by column name
            ncols: Number of columns in grid
            figsize: Figure size
            plot_type: 'box', 'violin', 'strip', or 'bar'
            sharey: Share y-axis across panels
            max_sig_bars: Maximum significance bars to show per panel
            log_scale: Whether to log10 transform the data for plotting
            show_points: Whether to overlay individual data points
            ylabel: Label for y-axis (e.g., 'Concentration', 'Ratio', '% of Total')
        """
        # Filter out NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))
        
        if figsize is None:
            figsize = (ncols * 4, nrows * 4)  # Slightly taller for significance bars
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Get valid groups only
        groups = [g for g in data[group_col].unique() if pd.notna(g)]
        palette = dict(zip(groups, self.group_palette[:len(groups)]))
        
        for i, col in enumerate(value_cols):
            if col not in data.columns:
                continue
            
            ax = axes[i]
            
            # Prepare plot data - apply log10 transform if requested
            plot_data = data.copy()
            plot_col = col

            if log_scale:
                # Create log10 transformed column, handling zeros
                log_col = f'{col}_log10'
                # Replace zeros/negatives with small positive value before log
                min_positive = plot_data[col][plot_data[col] > 0].min()
                floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
                plot_data[log_col] = np.log10(plot_data[col].clip(lower=floor_val))
                plot_col = log_col
            
            if plot_type == "box":
                sns.boxplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                          hue=group_col, palette=palette, order=groups, legend=False)
                if show_points:
                    sns.stripplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                                color='black', alpha=0.7, size=6, order=groups, 
                                jitter=0.2, zorder=10, edgecolor='white', linewidth=0.5)
            elif plot_type == "violin":
                sns.violinplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=palette, order=groups, 
                             inner=None, legend=False)  # Remove inner box to show points better
                if show_points:
                    sns.stripplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                                color='black', alpha=0.7, size=6, order=groups,
                                jitter=0.15, zorder=10, edgecolor='white', linewidth=0.5)
            elif plot_type == "strip":
                sns.stripplot(data=plot_data, x=group_col, y=plot_col, ax=ax,
                             hue=group_col, palette=palette, order=groups,
                             size=6, alpha=0.7, legend=False)
                # Add mean ± SEM lines
                grp_means = plot_data.groupby(group_col)[plot_col].mean()
                grp_sems = plot_data.groupby(group_col)[plot_col].sem()
                for j, g in enumerate(groups):
                    if g in grp_means.index:
                        ax.hlines(grp_means[g], j - 0.25, j + 0.25, color='black', linewidth=1.5, zorder=5)
                        ax.errorbar(j, grp_means[g], yerr=grp_sems[g], color='black',
                                   capsize=5, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)
            elif plot_type == "bar":
                means = plot_data.groupby(group_col)[plot_col].mean()
                sems = plot_data.groupby(group_col)[plot_col].sem()
                x = range(len(groups))
                ax.bar(x, [means[g] for g in groups],
                      yerr=[sems[g] for g in groups],
                      capsize=3, color=[palette[g] for g in groups],
                      edgecolor='black', linewidth=0.5)
                if show_points:
                    for j, g in enumerate(groups):
                        group_vals = plot_data[plot_data[group_col] == g][plot_col]
                        jitter = np.random.normal(0, 0.1, len(group_vals))
                        ax.scatter(j + jitter, group_vals, color='black', alpha=0.7, s=30, 
                                  zorder=10, edgecolors='white', linewidths=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(groups)
            
            # Add significance annotations (use original data for stats)
            if col in stats_results:
                result = stats_results[col]
                sig_pairs = self._get_significant_pairs(result, max_pairs=max_sig_bars)
                if sig_pairs:
                    self._add_significance_bars(ax, groups, sig_pairs, plot_data, plot_col, group_col)
                
                # Add p-value to title
                p = result.main_test.pvalue
                sig_marker = '*' if result.main_test.significant else ''
                ax.set_title(f"{col} (p={p:.3f}){sig_marker}", fontsize=10)
            else:
                ax.set_title(col, fontsize=10)
            
            ax.set_xlabel('')
            if log_scale:
                ax.set_ylabel(f'log₁₀({ylabel})')
            else:
                ax.set_ylabel(ylabel)

            ax.tick_params(axis='x', rotation=45)
        
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _get_significant_pairs(self, result, max_pairs: int = 3) -> List[Tuple[str, str, str]]:
        """Extract significant pairs from statistical result."""
        if not result.posthoc_test or result.posthoc_test.pairwise_results is None:
            return []
        
        pairs = []
        posthoc_df = result.posthoc_test.pairwise_results
        sig_rows = posthoc_df[posthoc_df['significant']].copy()
        
        # Sort by p-value to get most significant first
        p_col = 'pvalue_adj' if 'pvalue_adj' in sig_rows.columns else 'pvalue'
        sig_rows = sig_rows.sort_values(p_col)
        
        for _, row in sig_rows.head(max_pairs).iterrows():
            p = row.get('pvalue_adj', row.get('pvalue', 1.0))
            if p < 0.001:
                annot = '***'
            elif p < 0.01:
                annot = '**'
            elif p < 0.05:
                annot = '*'
            else:
                continue
            pairs.append((row['group1'], row['group2'], annot))
        
        return pairs
    
    def plot_grouped_comparison_with_stats(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_cols: List[str],
        value_labels: List[str],
        stats_results: Dict,
        title: str = '',
        ylabel: str = 'Concentration',
        plot_type: str = 'bar',
        figsize: Optional[Tuple[float, float]] = None,
        log_scale: bool = False
    ) -> Tuple[plt.Figure, plt.Axes, List[str]]:
        """
        Create grouped comparison plot with significance annotations for each value type.
        
        Shows multiple value columns (e.g., Ceramides/SM) grouped by
        the group column, with significance brackets comparing groups within each type.
        
        Args:
            data: DataFrame with group column and value columns
            group_col: Column name for grouping (x-axis groups)
            value_cols: List of column names to plot
            value_labels: Display labels for each value column
            stats_results: Dict mapping value_col names to StatisticalResult objects
            title: Plot title
            ylabel: Y-axis label
            plot_type: 'bar', 'box', 'strip', or 'violin'
            figsize: Figure size
            log_scale: Whether to log10 transform the data
            
        Returns:
            Tuple of (figure, axes, stats_text_list)
        """
        if figsize is None:
            figsize = (10, 7)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.strip().str.lower() != 'nan']
        
        # Apply log10 transform if requested
        plot_ylabel = ylabel
        if log_scale:
            for col in value_cols:
                min_positive = data[col][data[col] > 0].min()
                floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
                data[col] = np.log10(data[col].clip(lower=floor_val))
            plot_ylabel = f'log₁₀({ylabel})'
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        n_types = len(value_cols)
        
        # Colors for each type
        type_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'][:n_types]
        
        # Reshape data for seaborn
        # Rename columns to avoid conflicts with 'Type' var_name
        plot_data = data[[group_col] + value_cols].copy()
        plot_data = plot_data.rename(columns={group_col: '_Group_'})
        plot_data = plot_data.melt(
            id_vars='_Group_', 
            value_vars=value_cols,
            var_name='_Type_', 
            value_name='Value'
        )
        # Replace column names with labels
        label_map = dict(zip(value_cols, value_labels))
        plot_data['_Type_'] = plot_data['_Type_'].replace(label_map)
        
        # Create palette mapping labels to colors
        palette = dict(zip(value_labels, type_colors))
        
        # Plot based on type
        if plot_type == 'box':
            sns.boxplot(data=plot_data, x='_Group_', y='Value', hue='_Type_',
                       ax=ax, palette=palette, order=groups)
        elif plot_type == 'violin':
            sns.violinplot(data=plot_data, x='_Group_', y='Value', hue='_Type_',
                          ax=ax, palette=palette, order=groups, inner='box')
        elif plot_type == 'strip':
            sns.stripplot(data=plot_data, x='_Group_', y='Value', hue='_Type_',
                         ax=ax, palette=palette, order=groups, size=6, alpha=0.7)
            # Add mean ± SEM lines for each type within each group
            x = np.arange(n_groups)
            width = 0.8 / n_types
            for t_idx, label in enumerate(value_labels):
                offset = (t_idx - n_types / 2 + 0.5) * width
                type_data = plot_data[plot_data['_Type_'] == label]
                for g_idx, group in enumerate(groups):
                    cell = type_data[type_data['_Group_'] == group]['Value']
                    if len(cell) > 0:
                        m, s = cell.mean(), cell.sem()
                        xpos = x[g_idx] + offset
                        ax.hlines(m, xpos - width * 0.35, xpos + width * 0.35,
                                 color='black', linewidth=1.5, zorder=5)
                        ax.errorbar(xpos, m, yerr=s, color='black',
                                   capsize=4, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)
        else:  # bar
            sns.barplot(data=plot_data, x='_Group_', y='Value', hue='_Type_',
                       ax=ax, palette=palette, order=groups, capsize=0.1)
        
        ax.set_xlabel('')
        ax.set_ylabel(plot_ylabel)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Type', loc='upper right')
        
        # Add significance annotations for each type
        y_max = ax.get_ylim()[1]
        annotation_y_start = y_max * 1.02
        bar_height = y_max * 0.04
        
        stats_text_parts = []
        
        # Calculate x positions for each type within groups
        # For grouped plots, positions depend on number of types
        x = np.arange(n_groups)
        width = 0.8 / n_types
        
        for type_idx, (col, label) in enumerate(zip(value_cols, value_labels)):
            if col not in stats_results:
                continue
            
            result = stats_results[col]
            sig_pairs = self._get_significant_pairs(result, max_pairs=2)
            
            # Calculate offset for this type
            offset = (type_idx - n_types/2 + 0.5) * width
            
            # Add significance brackets
            for pair_idx, (g1, g2, annot) in enumerate(sig_pairs):
                if g1 in groups and g2 in groups:
                    idx1, idx2 = groups.index(g1), groups.index(g2)
                    x1, x2 = x[idx1] + offset, x[idx2] + offset
                    
                    # Stagger heights for multiple comparisons
                    y = annotation_y_start + (type_idx * 2 + pair_idx) * bar_height * 1.5
                    
                    # Draw bracket
                    ax.plot([x1, x1, x2, x2], [y, y + bar_height*0.3, y + bar_height*0.3, y],
                           color=type_colors[type_idx], linewidth=1.5)
                    ax.text((x1 + x2) / 2, y + bar_height*0.4, annot,
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           color=type_colors[type_idx])
            
            # Build stats text
            test_name = result.main_test.test_type.value
            p_val = result.main_test.pvalue
            sig_marker = "✔" if result.main_test.significant else ""
            stats_text_parts.append(f"{label}: {test_name} p={p_val:.4f} {sig_marker}")
        
        # Adjust y limit to accommodate annotations
        new_y_max = annotation_y_start + (n_types * 2 + 1) * bar_height * 1.5
        ax.set_ylim(top=new_y_max)
        
        plt.tight_layout()
        
        return fig, ax, stats_text_parts
    
    # Keep old name as alias for backwards compatibility
    def plot_grouped_bar_with_stats(self, *args, **kwargs):
        """Alias for plot_grouped_comparison_with_stats (backwards compatibility)."""
        return self.plot_grouped_comparison_with_stats(*args, **kwargs)
    
    def plot_composition_pie_charts(
        self,
        data: pd.DataFrame,
        group_col: str,
        pct_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        top_n: int = 10,
        other_threshold: float = 2.0
    ) -> plt.Figure:
        """
        Create pie charts showing sphingolipid pool composition for each group.
        
        Args:
            data: DataFrame with group column and percentage columns
            group_col: Column name for grouping
            pct_cols: List of percentage column names (with _pct suffix)
            title: Overall figure title
            figsize: Figure size (auto-calculated if None)
            top_n: Maximum number of slices to show (rest grouped as "Other")
            other_threshold: Minimum percentage to show individually (below this goes to "Other")
            
        Returns:
            Matplotlib figure with pie charts
        """
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.lower() != 'nan']
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        
        # Calculate figure size
        if figsize is None:
            figsize = (5 * n_groups, 5)
        
        fig, axes = plt.subplots(1, n_groups, figsize=figsize, dpi=150)
        if n_groups == 1:
            axes = [axes]
        
        # Use the canonical species->color map so colors are consistent across
        # experiments (not reassigned by per-experiment ranking).
        all_species = [col.replace('_pct', '') for col in pct_cols]
        color_map = {sp: self.species_color_map.get(sp, '#999999') for sp in all_species}
        color_map['Other'] = self.species_color_map.get('Other', '#F5F0E8')
        
        for idx, (group, ax) in enumerate(zip(groups, axes)):
            group_data = data[data[group_col] == group][pct_cols].mean()
            
            # Clean up names (remove _pct suffix)
            group_data.index = [col.replace('_pct', '') for col in group_data.index]
            
            # Sort by value and handle small slices
            group_data = group_data.sort_values(ascending=False)
            
            # Group small values into "Other"
            main_slices = group_data[group_data >= other_threshold].head(top_n)
            other_value = group_data.sum() - main_slices.sum()
            
            if other_value > 0.1:  # Only add "Other" if meaningful
                plot_data = pd.concat([main_slices, pd.Series({'Other': other_value})])
            else:
                plot_data = main_slices
            
            # Get colors for this pie
            pie_colors = [color_map.get(sp, '#999999') for sp in plot_data.index]

            # Build slice labels from stored values (not matplotlib's renormalized autopct),
            # so labels match the mean-of-per-sample-% shown in the bar/stat plots even when
            # rounded per-row percentages don't sum to exactly 100.
            slice_labels = [
                f'{val:.1f}%' if val >= 3 else ''
                for val in plot_data.values
            ]

            wedges, texts = ax.pie(
                plot_data.values,
                labels=None,  # legend added below
                colors=pie_colors,
                startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )

            total = plot_data.values.sum()
            pctdistance = 0.75
            for wedge, label in zip(wedges, slice_labels):
                if not label:
                    continue
                theta = np.deg2rad((wedge.theta2 + wedge.theta1) / 2)
                x = pctdistance * np.cos(theta)
                y = pctdistance * np.sin(theta)
                ax.text(x, y, label, ha='center', va='center',
                        fontsize=8, fontweight='bold')
            
            ax.set_title(f'{group}', fontsize=12, fontweight='bold')
        
        # Add a single legend for the whole figure
        # Get unique labels across all pies
        all_labels = set()
        for group in groups:
            group_data = data[data[group_col] == group][pct_cols].mean()
            group_data.index = [col.replace('_pct', '') for col in group_data.index]
            group_data = group_data.sort_values(ascending=False)
            main_slices = group_data[group_data >= other_threshold].head(top_n)
            all_labels.update(main_slices.index.tolist())
        
        # Sort labels by overall mean
        overall_means = data[pct_cols].mean()
        overall_means.index = [col.replace('_pct', '') for col in overall_means.index]
        sorted_labels = [l for l in overall_means.sort_values(ascending=False).index if l in all_labels]
        if 'Other' not in sorted_labels:
            sorted_labels.append('Other')
        
        # Create legend patches
        legend_patches = [mpatches.Patch(color=color_map.get(label, '#999999'), label=label) 
                         for label in sorted_labels if label in all_labels or label == 'Other']
        
        fig.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5),
                  fontsize=9, title='Sphingolipids')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig

    def plot_category_composition_pie_charts(
        self,
        data: pd.DataFrame,
        group_col: str,
        sphingolipid_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        other_threshold: float = 2.0
    ) -> plt.Figure:
        """
        Create pie charts showing within-category sphingolipid composition.

        One column per category (ceramides, dihydroceramides, sphingomyelins,
        hexosylceramides, ceramide-1-phosphates, sphingoid bases, sphingoid base
        phosphates); one row per group. Slices show individual species percentages
        within that category.
        """
        from config.sphingolipid_species import (
            get_ceramides, get_dihydroceramides, get_sphingomyelins,
            get_hexosylceramides, get_ceramide_1_phosphates,
            get_sphingoid_bases, get_sphingoid_base_phosphates,
        )

        available = set(sphingolipid_cols) & set(data.columns)

        categories = {
            'Ceramides': [s for s in get_ceramides() if s in available],
            'Dihydroceramides': [s for s in get_dihydroceramides() if s in available],
            'Sphingomyelins': [s for s in get_sphingomyelins() if s in available],
            'Hexosylceramides': [s for s in get_hexosylceramides() if s in available],
            'Ceramide-1-Phosphates': [s for s in get_ceramide_1_phosphates() if s in available],
            'Sphingoid Bases': [s for s in get_sphingoid_bases() if s in available],
            'Sphingoid Base Phosphates': [s for s in get_sphingoid_base_phosphates() if s in available],
        }
        categories = {k: v for k, v in categories.items() if v}

        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.lower() != 'nan']

        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        n_categories = len(categories)

        if n_categories == 0:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(0.5, 0.5, 'No category data available', ha='center', va='center')
            return fig

        if figsize is None:
            figsize = (5 * n_categories, 5.5 * n_groups)

        fig, axes = plt.subplots(n_groups, n_categories, figsize=figsize,
                                 squeeze=False, dpi=150)
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Pull colors from the instance-wide canonical species map so the same
        # species gets the same color in every experiment.
        sp_color_map = dict(self.species_color_map)

        category_names = list(categories.keys())

        for row_idx, group in enumerate(groups):
            group_data = data[data[group_col] == group]

            for col_idx, cat_name in enumerate(category_names):
                ax = axes[row_idx, col_idx]
                species_list = categories[cat_name]

                means = group_data[species_list].mean()
                cat_total = means.sum()

                if cat_total <= 0 or pd.isna(cat_total):
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
                    ax.set_title(f'{cat_name}', fontsize=10, fontweight='bold')
                    if col_idx == 0:
                        ax.set_ylabel(f'{group}', fontsize=11, fontweight='bold')
                    continue

                pcts = (means / cat_total * 100).sort_values(ascending=False)

                main_slices = pcts[pcts >= other_threshold]
                other_value = pcts[pcts < other_threshold].sum()

                if other_value > 0.1:
                    plot_data = pd.concat([main_slices, pd.Series({'Other': other_value})])
                else:
                    plot_data = main_slices

                pie_colors = [sp_color_map.get(sp, '#F5F0E8') for sp in plot_data.index]

                # Label each wedge with its stored percentage (no autopct renormalization)
                slice_labels = [
                    f'{val:.1f}%' if val >= 3 else ''
                    for val in plot_data.values
                ]

                wedges, _texts = ax.pie(
                    plot_data.values,
                    labels=None,
                    colors=pie_colors,
                    startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                )

                pctdistance = 0.75
                for wedge, label in zip(wedges, slice_labels):
                    if not label:
                        continue
                    theta = np.deg2rad((wedge.theta2 + wedge.theta1) / 2)
                    x = pctdistance * np.cos(theta)
                    y = pctdistance * np.sin(theta)
                    ax.text(x, y, label, ha='center', va='center',
                            fontsize=12, fontweight='bold')

                if row_idx == 0:
                    ax.set_title(f'{cat_name}', fontsize=14, fontweight='bold', pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f'{group}', fontsize=14, fontweight='bold', labelpad=15)

        # Per-category legends along the bottom row
        for col_idx, cat_name in enumerate(category_names):
            species_list = categories[cat_name]
            displayed = set()
            any_nonzero = False
            for group in groups:
                gd = data[data[group_col] == group]
                means = gd[species_list].mean()
                cat_total = means.sum()
                if cat_total > 0:
                    any_nonzero = True
                    pcts = means / cat_total * 100
                    displayed.update(pcts[pcts >= other_threshold].index.tolist())

            sorted_sps = sorted(displayed)
            legend_items = [mpatches.Patch(color=sp_color_map.get(sp, '#F5F0E8'), label=sp)
                            for sp in sorted_sps]
            if any_nonzero:
                legend_items.append(mpatches.Patch(color='#F5F0E8', label='Other'))

            axes[-1, col_idx].legend(
                handles=legend_items, loc='upper center',
                bbox_to_anchor=(0.5, -0.08), fontsize=10, ncol=2,
                frameon=True, fancybox=True
            )

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        return fig

    def plot_composition_horizontal_bars(
        self,
        data: pd.DataFrame,
        group_col: str,
        pct_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        top_n: int = 15,
        show_values: bool = True
    ) -> plt.Figure:
        """
        Create horizontal bar charts showing sphingolipid percentages by group.
        
        Args:
            data: DataFrame with group column and percentage columns
            group_col: Column name for grouping
            pct_cols: List of percentage column names (with _pct suffix)
            title: Overall figure title
            figsize: Figure size (auto-calculated if None)
            top_n: Maximum number of sphingolipids to show
            show_values: Whether to show percentage values on bars
            
        Returns:
            Matplotlib figure with horizontal bar charts
        """
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.lower() != 'nan']
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        
        # Calculate mean percentages for each group
        group_means = data.groupby(group_col)[pct_cols].mean()
        
        # Clean column names (remove _pct suffix)
        group_means.columns = [col.replace('_pct', '') for col in group_means.columns]
        
        # Get top N sphingolipids by overall mean
        overall_means = group_means.mean().sort_values(ascending=False)
        top_species = overall_means.head(top_n).index.tolist()
        
        # Calculate figure size
        if figsize is None:
            figsize = (12, max(6, len(top_species) * 0.4))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up positions
        y_pos = np.arange(len(top_species))
        bar_height = 0.8 / n_groups
        
        # Get colors for groups
        group_colors = dict(zip(groups, self.group_palette[:n_groups]))
        
        # Plot bars for each group
        for i, group in enumerate(groups):
            values = [group_means.loc[group, sp] for sp in top_species]
            offset = (i - n_groups/2 + 0.5) * bar_height
            
            bars = ax.barh(y_pos + offset, values, bar_height * 0.9,
                          label=group, color=group_colors[group],
                          edgecolor='white', linewidth=0.5)
            
            # Add value labels
            if show_values:
                for bar, val in zip(bars, values):
                    if val >= 1:  # Only show if >= 1%
                        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                               f'{val:.1f}%', va='center', ha='left', fontsize=8)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_species)
        ax.invert_yaxis()  # Top sphingolipid at top
        ax.set_xlabel('% of Total Sphingolipid Pool')
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)  # Add space for labels
        
        ax.legend(title='Group', loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_composition_stacked_horizontal(
        self,
        data: pd.DataFrame,
        group_col: str,
        pct_cols: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        top_n: int = 12
    ) -> plt.Figure:
        """
        Create stacked horizontal bar chart showing sphingolipid pool composition.
        Each bar represents a group, segments show sphingolipid proportions.
        
        Args:
            data: DataFrame with group column and percentage columns
            group_col: Column name for grouping
            pct_cols: List of percentage column names (with _pct suffix)
            title: Overall figure title
            figsize: Figure size
            top_n: Maximum number of sphingolipids to show (rest grouped as "Other")
            
        Returns:
            Matplotlib figure
        """
        # Filter NaN groups
        data = data.copy()
        data = data[data[group_col].notna()]
        data = data[data[group_col].astype(str).str.lower() != 'nan']
        
        groups = data[group_col].unique().tolist()
        n_groups = len(groups)
        
        # Calculate mean percentages for each group
        group_means = data.groupby(group_col)[pct_cols].mean()
        group_means.columns = [col.replace('_pct', '') for col in group_means.columns]
        
        # Get top N sphingolipids by overall mean
        overall_means = group_means.mean().sort_values(ascending=False)
        top_species = overall_means.head(top_n).index.tolist()
        
        # Create "Other" category
        other_species = [sp for sp in group_means.columns if sp not in top_species]
        if other_species:
            group_means['Other'] = group_means[other_species].sum(axis=1)
            plot_species = top_species + ['Other']
        else:
            plot_species = top_species
        
        # Calculate figure size
        if figsize is None:
            figsize = (12, max(4, n_groups * 1.2))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_species)))
        
        # Plot stacked horizontal bars
        y_pos = np.arange(n_groups)
        left = np.zeros(n_groups)
        
        for i, sp in enumerate(plot_species):
            values = group_means.loc[groups, sp].values
            bars = ax.barh(y_pos, values, left=left, label=sp, color=colors[i],
                          edgecolor='white', linewidth=0.5, height=0.6)
            
            # Add labels for segments >= 5%
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val >= 5:
                    ax.text(left[j] + val/2, bar.get_y() + bar.get_height()/2,
                           f'{val:.0f}%', va='center', ha='center', 
                           fontsize=8, fontweight='bold', color='white')
            
            left += values
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups)
        ax.set_xlabel('% of Total Sphingolipid Pool')
        ax.set_xlim(0, 100)
        
        # Legend outside
        ax.legend(title='Sphingolipids', bbox_to_anchor=(1.02, 1), loc='upper left',
                 fontsize=8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    # ================================================================
    # TWO-WAY ANOVA VISUALIZATION (completely separate from one-way)
    # ================================================================
    
    def plot_twoway_bar(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        twoway_result=None,
        ax: Optional[plt.Axes] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
        show_points: bool = True,
        title: Optional[str] = None,
        show_anova_text: bool = True,
        plot_type: str = "bar",
    ) -> plt.Axes:
        """
        Create grouped plot for two-way ANOVA.

        Factor A on x-axis, Factor B as hue/color with legend.
        Supports bar, box, strip, and violin plot types.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_single)

        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col

        # Clean data
        df = data[[value_col, factor_a_col, factor_b_col]].dropna().copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()

        # Get ordered levels
        a_levels = sorted(df[factor_a_col].unique())
        b_levels = sorted(df[factor_b_col].unique())
        n_a = len(a_levels)
        n_b = len(b_levels)

        # Bar positioning (used for bar plot and significance brackets)
        bar_width = 0.8 / n_b
        x_base = np.arange(n_a)
        palette = self.group_palette[:n_b]
        palette_dict = dict(zip(b_levels, palette))

        if plot_type == "box":
            sns.boxplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                       ax=ax, palette=palette_dict, order=a_levels, hue_order=b_levels)
            if show_points:
                sns.stripplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                             ax=ax, dodge=True, color='black', alpha=0.6, size=4,
                             order=a_levels, hue_order=b_levels, legend=False,
                             jitter=0.15, zorder=10, edgecolor='white', linewidth=0.3)
        elif plot_type == "violin":
            sns.violinplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                          ax=ax, palette=palette_dict, order=a_levels, hue_order=b_levels,
                          inner=None)
            if show_points:
                sns.stripplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                             ax=ax, dodge=True, color='black', alpha=0.6, size=4,
                             order=a_levels, hue_order=b_levels, legend=False,
                             jitter=0.15, zorder=10, edgecolor='white', linewidth=0.3)
        elif plot_type == "strip":
            sns.stripplot(data=df, x=factor_a_col, y=value_col, hue=factor_b_col,
                         ax=ax, dodge=True, palette=palette_dict, order=a_levels, hue_order=b_levels,
                         size=4, alpha=0.6)
            # Add mean ± SEM lines for each factor_b level within each factor_a level
            for j, b_level in enumerate(b_levels):
                x_pos = x_base + (j - (n_b - 1) / 2) * bar_width
                for i_a, a_level in enumerate(a_levels):
                    cell = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                    if len(cell) > 0:
                        m, s = cell.mean(), cell.sem() if len(cell) > 1 else 0
                        ax.hlines(m, x_pos[i_a] - bar_width * 0.35, x_pos[i_a] + bar_width * 0.35,
                                 color='black', linewidth=1.5, zorder=5)
                        ax.errorbar(x_pos[i_a], m, yerr=s, color='black',
                                   capsize=4, capthick=1.5, linewidth=1.5, fmt='none', zorder=5)
        else:  # "bar"
            for j, b_level in enumerate(b_levels):
                x_pos = x_base + (j - (n_b - 1) / 2) * bar_width
                means = []
                sems = []
                for a_level in a_levels:
                    cell = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                    means.append(cell.mean() if len(cell) > 0 else 0)
                    sems.append(cell.sem() if len(cell) > 1 else 0)

                ax.bar(x_pos, means, width=bar_width * 0.9, yerr=sems,
                       capsize=3, color=palette[j], edgecolor='black', linewidth=0.5,
                       label=str(b_level), zorder=2)

                if show_points:
                    for i, a_level in enumerate(a_levels):
                        cell_vals = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                        if len(cell_vals) > 0:
                            jitter = np.random.normal(0, bar_width * 0.08, len(cell_vals))
                            ax.scatter(x_pos[i] + jitter, cell_vals, color='black', alpha=0.6,
                                       s=20, zorder=10, edgecolors='white', linewidths=0.3)

            ax.set_xticks(x_base)
            ax.set_xticklabels(a_levels)

        ax.set_xlabel(fa_name)
        ax.set_ylabel(ylabel)

        # Add post-hoc significance brackets FIRST (they expand y-axis)
        if twoway_result is not None and twoway_result.posthoc_results is not None:
            self._add_twoway_significance_bars(
                ax, df, value_col, factor_a_col, factor_b_col,
                a_levels, b_levels, bar_width, twoway_result
            )

        # Add ANOVA p-value text annotation AFTER brackets so it doesn't overlap
        if twoway_result is not None and show_anova_text:
            tw = twoway_result
            import math
            def _stars(p):
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    return 'N/A'
                if p < 0.001: return '***'
                if p < 0.01: return '**'
                if p < 0.05: return '*'
                return 'n.s.'
            def _fmt_p(p):
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    return 'N/A'
                return f"{p:.3f}"

            anova_text = (
                f"{tw.factor_a_name}: p={_fmt_p(tw.factor_a_pvalue)} {_stars(tw.factor_a_pvalue)}\n"
                f"{tw.factor_b_name}: p={_fmt_p(tw.factor_b_pvalue)} {_stars(tw.factor_b_pvalue)}\n"
                f"{tw.factor_a_name}×{tw.factor_b_name}: p={_fmt_p(tw.interaction_pvalue)} {_stars(tw.interaction_pvalue)}"
            )
            # Add padding above brackets for the text box
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(top=y_hi + (y_hi - y_lo) * 0.15)
            ax.text(0.02, 0.98, anova_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

        # Legend always top-right, outside data area
        ax.legend(title=fb_name, loc='upper right', framealpha=0.9)
        
        if title:
            ax.set_title(title, fontsize=10)
        elif twoway_result is not None:
            # Show interaction significance in title
            p_int = twoway_result.interaction_pvalue
            sig_marker = '*' if twoway_result.interaction_significant else ''
            p_str = 'N/A' if (p_int is None or (isinstance(p_int, float) and math.isnan(p_int))) else f"{p_int:.3f}"
            ax.set_title(f"{value_col} (interaction p={p_str}){sig_marker}", fontsize=10)
        else:
            ax.set_title(value_col, fontsize=10)
        
        return ax
    
    def _add_twoway_significance_bars(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        a_levels: list,
        b_levels: list,
        bar_width: float,
        twoway_result,
        max_bars: int = 4
    ):
        """Add significance bracket annotations for two-way post-hoc comparisons."""
        posthoc = twoway_result.posthoc_results
        if posthoc is None or len(posthoc) == 0:
            return
        
        sig_rows = posthoc[posthoc['significant']].copy()
        p_col = 'pvalue_adj' if 'pvalue_adj' in sig_rows.columns else 'pvalue'
        sig_rows = sig_rows.sort_values(p_col).head(max_bars)
        
        if len(sig_rows) == 0:
            return
        
        # Calculate y positions for brackets
        y_max = ax.get_ylim()[1]
        y_step = (y_max - ax.get_ylim()[0]) * 0.06
        current_y = y_max
        
        n_b = len(b_levels)
        x_base = np.arange(len(a_levels))
        
        for _, row in sig_rows.iterrows():
            p = row.get('pvalue_adj', row.get('pvalue', 1.0))
            
            g1, g2 = str(row['group1']), str(row['group2'])
            within_level = str(row.get('within_level', 'all'))
            factor_compared = row.get('factor_compared', '')
            
            # For main effect marginal comparisons, use the ANOVA main effect
            # p-value rather than the post-hoc test p-value, since the omnibus
            # F-test is the definitive result for the main effect.
            if twoway_result.posthoc_type == "main_effect_marginal":
                if factor_compared == twoway_result.factor_a_name:
                    anova_p = twoway_result.factor_a_pvalue
                    if anova_p is not None and not (isinstance(anova_p, float) and math.isnan(anova_p)):
                        p = anova_p
                elif factor_compared == twoway_result.factor_b_name:
                    anova_p = twoway_result.factor_b_pvalue
                    if anova_p is not None and not (isinstance(anova_p, float) and math.isnan(anova_p)):
                        p = anova_p
            
            if p < 0.001:
                annot = '***'
            elif p < 0.01:
                annot = '**'
            elif p < 0.05:
                annot = '*'
            else:
                continue
            
            # Determine x positions for the bracket
            try:
                if twoway_result.posthoc_type == "simple_main_effects":
                    if factor_compared == twoway_result.factor_a_name:
                        # Comparing Factor A levels within a Factor B level
                        b_idx = b_levels.index(within_level) if within_level in b_levels else 0
                        a_idx1 = a_levels.index(g1) if g1 in a_levels else None
                        a_idx2 = a_levels.index(g2) if g2 in a_levels else None
                        if a_idx1 is None or a_idx2 is None:
                            continue
                        x1 = x_base[a_idx1] + (b_idx - (n_b - 1) / 2) * bar_width
                        x2 = x_base[a_idx2] + (b_idx - (n_b - 1) / 2) * bar_width
                    else:
                        # Comparing Factor B levels within a Factor A level
                        a_idx = a_levels.index(within_level) if within_level in a_levels else 0
                        b_idx1 = b_levels.index(g1) if g1 in b_levels else None
                        b_idx2 = b_levels.index(g2) if g2 in b_levels else None
                        if b_idx1 is None or b_idx2 is None:
                            continue
                        x1 = x_base[a_idx] + (b_idx1 - (n_b - 1) / 2) * bar_width
                        x2 = x_base[a_idx] + (b_idx2 - (n_b - 1) / 2) * bar_width
                else:
                    # Main effect marginal comparisons
                    if factor_compared == twoway_result.factor_a_name:
                        a_idx1 = a_levels.index(g1) if g1 in a_levels else None
                        a_idx2 = a_levels.index(g2) if g2 in a_levels else None
                        if a_idx1 is None or a_idx2 is None:
                            continue
                        x1 = x_base[a_idx1]
                        x2 = x_base[a_idx2]
                    else:
                        # Factor B marginal - skip bracket (shown in text annotation)
                        continue
            except (ValueError, IndexError):
                continue
            
            # Draw bracket
            bracket_y = current_y
            ax.plot([x1, x1, x2, x2], 
                    [bracket_y, bracket_y + y_step * 0.3, bracket_y + y_step * 0.3, bracket_y],
                    color='black', linewidth=0.8)
            ax.text((x1 + x2) / 2, bracket_y + y_step * 0.35, annot,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            current_y += y_step
        
        # Expand y-axis to fit brackets
        ax.set_ylim(top=current_y + y_step)
    
    def plot_twoway_interaction(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        twoway_result=None,
        ax: Optional[plt.Axes] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
    ) -> plt.Axes:
        """
        Create interaction plot (line plot) for two-way ANOVA.
        
        Factor A on x-axis, separate lines for each Factor B level.
        Shows means ± SEM. Parallel lines = no interaction, 
        crossing/diverging lines = interaction present.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_single)
        
        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col
        
        df = data[[value_col, factor_a_col, factor_b_col]].dropna().copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()
        
        a_levels = sorted(df[factor_a_col].unique())
        b_levels = sorted(df[factor_b_col].unique())
        palette = self.group_palette[:len(b_levels)]
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        x_pos = np.arange(len(a_levels))
        
        for j, b_level in enumerate(b_levels):
            means = []
            sems = []
            for a_level in a_levels:
                cell = df[(df[factor_a_col] == a_level) & (df[factor_b_col] == b_level)][value_col]
                means.append(cell.mean() if len(cell) > 0 else np.nan)
                sems.append(cell.sem() if len(cell) > 1 else 0)
            
            marker = markers[j % len(markers)]
            ax.errorbar(x_pos, means, yerr=sems, 
                       marker=marker, markersize=8, 
                       color=palette[j], linewidth=2,
                       capsize=4, capthick=1.5,
                       label=str(b_level), zorder=5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(a_levels)
        ax.set_xlabel(fa_name)
        ax.set_ylabel(ylabel)
        ax.legend(title=fb_name, loc='best')
        
        title = f"{value_col} — Interaction Plot"
        if twoway_result is not None:
            p_int = twoway_result.interaction_pvalue
            sig = "SIGNIFICANT" if twoway_result.interaction_significant else "n.s."
            import math
            p_str = 'N/A' if (p_int is None or (isinstance(p_int, float) and math.isnan(p_int))) else f"{p_int:.3f}"
            title += f"\n(interaction p={p_str}, {sig})"
        ax.set_title(title, fontsize=10)
        
        return ax
    
    def plot_twoway_multi_panel(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        factor_a_col: str,
        factor_b_col: str,
        twoway_results: Dict = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
        show_points: bool = True,
        plot_type: str = "bar",
    ) -> plt.Figure:
        """
        Create multi-panel two-way grouped plots with significance annotations.

        Equivalent to plot_multi_panel_groups_with_stats but for two-way designs.
        Each panel shows one analyte with Factor A on x-axis, Factor B as hue.
        """
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))

        if figsize is None:
            figsize = (ncols * 4.5, nrows * 4.5)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(value_cols):
            if col not in data.columns:
                axes[i].set_visible(False)
                continue

            tw_result = None
            if twoway_results and col in twoway_results:
                result_obj = twoway_results[col]
                # Handle both FullTwoWayAnalysisResult and TwoWayResult
                if hasattr(result_obj, 'twoway_result'):
                    tw_result = result_obj.twoway_result
                else:
                    tw_result = result_obj

            self.plot_twoway_bar(
                data=data, value_col=col,
                factor_a_col=factor_a_col, factor_b_col=factor_b_col,
                twoway_result=tw_result, ax=axes[i],
                factor_a_name=factor_a_name, factor_b_name=factor_b_name,
                ylabel=ylabel, show_points=show_points,
                plot_type=plot_type,
            )
        
        # Hide empty panels
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_twoway_interaction_multi_panel(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        factor_a_col: str,
        factor_b_col: str,
        twoway_results: Dict = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        ylabel: str = "Concentration",
    ) -> plt.Figure:
        """Multi-panel interaction (line) plots for two-way ANOVA."""
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))
        
        if figsize is None:
            figsize = (ncols * 4.5, nrows * 4)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(value_cols):
            if col not in data.columns:
                axes[i].set_visible(False)
                continue
            
            tw_result = None
            if twoway_results and col in twoway_results:
                result_obj = twoway_results[col]
                if hasattr(result_obj, 'twoway_result'):
                    tw_result = result_obj.twoway_result
                else:
                    tw_result = result_obj
            
            self.plot_twoway_interaction(
                data=data, value_col=col,
                factor_a_col=factor_a_col, factor_b_col=factor_b_col,
                twoway_result=tw_result, ax=axes[i],
                factor_a_name=factor_a_name, factor_b_name=factor_b_name,
                ylabel=ylabel,
            )
        
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    # ================================================================
    # THREE-WAY ANOVA VISUALIZATION 
    # ================================================================

    def plot_threeway_faceted(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        factor_c_col: str,
        threeway_result=None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        factor_c_name: Optional[str] = None,
        ylabel: str = "Concentration",
        show_points: bool = True,
        plot_type: str = "bar",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create faceted grouped plot for three-way ANOVA.

        One subplot per level of Factor C, with Factor A on x-axis
        and Factor B as hue/color.
        """
        import math

        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col
        fc_name = factor_c_name or factor_c_col

        # Clean data
        df = data[[value_col, factor_a_col, factor_b_col, factor_c_col]].dropna().copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()

        c_levels = sorted(df[factor_c_col].unique())
        n_c = len(c_levels)

        has_anova = threeway_result is not None
        fig_height = 6.5 if has_anova else 5
        fig, axes = plt.subplots(1, n_c, figsize=(max(5, 4 * n_c), fig_height), sharey=True)
        if n_c == 1:
            axes = [axes]

        a_levels = sorted(df[factor_a_col].unique())
        b_levels = sorted(df[factor_b_col].unique())
        n_a = len(a_levels)
        n_b = len(b_levels)

        bar_width = 0.8 / n_b
        x_base = np.arange(n_a)
        palette = self.group_palette[:n_b]

        for c_idx, c_level in enumerate(c_levels):
            ax = axes[c_idx]
            subset = df[df[factor_c_col] == c_level]

            if plot_type == "box":
                palette_dict = dict(zip(b_levels, palette))
                sns.boxplot(data=subset, x=factor_a_col, y=value_col, hue=factor_b_col,
                           ax=ax, palette=palette_dict, order=a_levels, hue_order=b_levels)
                if show_points:
                    sns.stripplot(data=subset, x=factor_a_col, y=value_col, hue=factor_b_col,
                                 ax=ax, dodge=True, palette='dark:black', alpha=0.6, size=3,
                                 order=a_levels, hue_order=b_levels, legend=False,
                                 jitter=0.15, zorder=10, edgecolor='white', linewidth=0.3)
            else:  # "bar"
                for j, b_level in enumerate(b_levels):
                    x_pos = x_base + (j - (n_b - 1) / 2) * bar_width
                    means = []
                    sems = []
                    for a_level in a_levels:
                        cell = subset[(subset[factor_a_col] == a_level) & (subset[factor_b_col] == b_level)][value_col]
                        means.append(cell.mean() if len(cell) > 0 else 0)
                        sems.append(cell.sem() if len(cell) > 1 else 0)

                    ax.bar(x_pos, means, width=bar_width * 0.9, yerr=sems,
                           capsize=3, color=palette[j], edgecolor='black', linewidth=0.5,
                           label=str(b_level) if c_idx == 0 else None, zorder=2)

                    if show_points:
                        for i, a_level in enumerate(a_levels):
                            cell_vals = subset[(subset[factor_a_col] == a_level) & (subset[factor_b_col] == b_level)][value_col]
                            if len(cell_vals) > 0:
                                jitter = np.random.normal(0, bar_width * 0.08, len(cell_vals))
                                ax.scatter(x_pos[i] + jitter, cell_vals, color='black', alpha=0.6,
                                           s=15, zorder=10, edgecolors='white', linewidths=0.3)

                ax.set_xticks(x_base)
                ax.set_xticklabels(a_levels)

            ax.set_xlabel(fa_name)
            if c_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f'{fc_name} = {c_level}', fontsize=10)

            # Only show legend on first panel
            if c_idx == 0:
                ax.legend(title=fb_name, loc='upper right', framealpha=0.9, fontsize=7)
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

        if title:
            fig.suptitle(title, fontsize=12, y=1.02)

        plt.tight_layout()

        # Add ANOVA summary text AFTER tight_layout to prevent clipping
        if threeway_result is not None:
            tw = threeway_result
            if hasattr(tw, 'threeway_result'):
                tw = tw.threeway_result
            def _stars(p):
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    return '?'
                if p < 0.001: return '***'
                if p < 0.01: return '**'
                if p < 0.05: return '*'
                return 'ns'
            def _fmt_p(p):
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    return 'N/A'
                if p < 0.001: return '<.001'
                return f"{p:.3f}"

            anova_text = (
                f"Main effects:    {tw.factor_a_name}: p={_fmt_p(tw.factor_a_pvalue)} {_stars(tw.factor_a_pvalue)}  |  "
                f"{tw.factor_b_name}: p={_fmt_p(tw.factor_b_pvalue)} {_stars(tw.factor_b_pvalue)}  |  "
                f"{tw.factor_c_name}: p={_fmt_p(tw.factor_c_pvalue)} {_stars(tw.factor_c_pvalue)}\n"
                f"2-way:  {tw.factor_a_name}\u00d7{tw.factor_b_name}: p={_fmt_p(tw.interaction_ab_pvalue)} {_stars(tw.interaction_ab_pvalue)}  |  "
                f"{tw.factor_a_name}\u00d7{tw.factor_c_name}: p={_fmt_p(tw.interaction_ac_pvalue)} {_stars(tw.interaction_ac_pvalue)}  |  "
                f"{tw.factor_b_name}\u00d7{tw.factor_c_name}: p={_fmt_p(tw.interaction_bc_pvalue)} {_stars(tw.interaction_bc_pvalue)}\n"
                f"3-way:  {tw.factor_a_name}\u00d7{tw.factor_b_name}\u00d7{tw.factor_c_name}: p={_fmt_p(tw.interaction_abc_pvalue)} {_stars(tw.interaction_abc_pvalue)}"
            )
            fig.subplots_adjust(bottom=0.18)
            fig.text(0.02, 0.01, anova_text, fontsize=7, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.9,
                               edgecolor='#c0a060'),
                     verticalalignment='bottom')

        return fig

    def plot_threeway_interaction(
        self,
        data: pd.DataFrame,
        value_col: str,
        factor_a_col: str,
        factor_b_col: str,
        factor_c_col: str,
        threeway_result=None,
        ax: Optional[plt.Axes] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        factor_c_name: Optional[str] = None,
        ylabel: str = "Concentration",
    ) -> plt.Figure:
        """
        Create faceted interaction plots for three-way ANOVA.

        One subplot per level of Factor C, with Factor A on x-axis
        and separate lines for Factor B levels.
        """
        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col
        fc_name = factor_c_name or factor_c_col

        df = data[[value_col, factor_a_col, factor_b_col, factor_c_col]].dropna().copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()

        c_levels = sorted(df[factor_c_col].unique())
        a_levels = sorted(df[factor_a_col].unique())
        b_levels = sorted(df[factor_b_col].unique())
        n_c = len(c_levels)

        fig, axes = plt.subplots(1, n_c, figsize=(max(5, 4 * n_c), 4), sharey=True)
        if n_c == 1:
            axes = [axes]

        palette = self.group_palette[:len(b_levels)]
        x_pos = np.arange(len(a_levels))

        for c_idx, c_level in enumerate(c_levels):
            ax = axes[c_idx]
            subset = df[df[factor_c_col] == c_level]

            for b_idx, b_level in enumerate(b_levels):
                means = []
                sems = []
                for a_level in a_levels:
                    cell = subset[(subset[factor_a_col] == a_level) & (subset[factor_b_col] == b_level)][value_col]
                    means.append(cell.mean() if len(cell) > 0 else np.nan)
                    sems.append(cell.sem() if len(cell) > 1 else 0)

                ax.errorbar(x_pos, means, yerr=sems, marker='o', capsize=4,
                           color=palette[b_idx], linewidth=1.5, markersize=6,
                           label=str(b_level) if c_idx == 0 else None)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(a_levels)
            ax.set_xlabel(fa_name)
            if c_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f'{fc_name} = {c_level}', fontsize=10)

            if c_idx == 0:
                ax.legend(title=fb_name, fontsize=7, framealpha=0.9)

        if threeway_result is not None:
            tw = threeway_result
            if hasattr(tw, 'threeway_result'):
                tw = tw.threeway_result
            fig.suptitle(f'{value_col} — {fa_name} \u00d7 {fb_name} \u00d7 {fc_name}', fontsize=11)

        plt.tight_layout()
        return fig

    def plot_threeway_multi_panel(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        factor_a_col: str,
        factor_b_col: str,
        factor_c_col: str,
        threeway_results: Dict = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        factor_c_name: Optional[str] = None,
        ylabel: str = "Concentration",
        show_points: bool = True,
        plot_type: str = "bar",
        log_scale: bool = False,
    ) -> plt.Figure:
        """
        Multi-panel faceted bar plots for three-way ANOVA.

        Each analyte gets a row of subplots (one per Factor C level)
        plus an ANOVA significance annotation column on the right.
        """
        import math

        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col
        fc_name = factor_c_name or factor_c_col

        def _stars(p):
            if p is None or (isinstance(p, float) and math.isnan(p)):
                return '?'
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return 'ns'

        def _fmt_p(p):
            if p is None or (isinstance(p, float) and math.isnan(p)):
                return 'N/A'
            if p < 0.001: return '<.001'
            return f'{p:.3f}'

        df = data.copy()
        c_levels = sorted(df[factor_c_col].astype(str).str.strip().unique())
        n_c = len(c_levels)
        n_analytes = len(value_cols)

        has_results = threeway_results is not None and len(threeway_results) > 0
        total_cols = n_c + (1 if has_results else 0)

        if figsize is None:
            plot_width = max(5, 4 * n_c) + (3.2 if has_results else 0)
            figsize = (plot_width, max(4, 3.5 * n_analytes))

        if has_results:
            fig, axes = plt.subplots(n_analytes, total_cols, figsize=figsize,
                                     gridspec_kw={'width_ratios': [1]*n_c + [0.65]},
                                     squeeze=False)
        else:
            fig, axes = plt.subplots(n_analytes, n_c, figsize=figsize, squeeze=False)

        # Share y-axis within each row (only among plot columns)
        for row_idx in range(n_analytes):
            for c_idx in range(1, n_c):
                axes[row_idx, c_idx].sharey(axes[row_idx, 0])

        a_levels = sorted(df[factor_a_col].astype(str).str.strip().unique())
        b_levels = sorted(df[factor_b_col].astype(str).str.strip().unique())
        n_a = len(a_levels)
        n_b = len(b_levels)
        bar_width = 0.8 / n_b
        x_base = np.arange(n_a)
        palette = self.group_palette[:n_b]

        for row_idx, col in enumerate(value_cols):
            if col not in df.columns:
                for c_idx in range(total_cols):
                    axes[row_idx, c_idx].set_visible(False)
                continue

            # Apply log10 transform if requested
            plot_col = col
            if log_scale:
                plot_col = f'{col}_log10'
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                min_positive = numeric_vals[numeric_vals > 0].min()
                floor_val = min_positive / 10 if pd.notna(min_positive) else 0.001
                df[plot_col] = np.log10(numeric_vals.clip(lower=floor_val))

            tw_result = None
            if threeway_results and col in threeway_results:
                result_obj = threeway_results[col]
                if hasattr(result_obj, 'threeway_result'):
                    tw_result = result_obj.threeway_result
                else:
                    tw_result = result_obj

            for c_idx, c_level in enumerate(c_levels):
                ax = axes[row_idx, c_idx]
                subset = df[df[factor_c_col].astype(str).str.strip() == c_level]

                for j, b_level in enumerate(b_levels):
                    x_pos = x_base + (j - (n_b - 1) / 2) * bar_width
                    means = []
                    sems = []
                    for a_level in a_levels:
                        cell = subset[(subset[factor_a_col].astype(str).str.strip() == a_level) &
                                      (subset[factor_b_col].astype(str).str.strip() == b_level)][plot_col]
                        cell = pd.to_numeric(cell, errors='coerce').dropna()
                        means.append(cell.mean() if len(cell) > 0 else 0)
                        sems.append(cell.sem() if len(cell) > 1 else 0)

                    ax.bar(x_pos, means, width=bar_width * 0.9, yerr=sems,
                           capsize=2, color=palette[j], edgecolor='black', linewidth=0.5,
                           label=str(b_level) if (row_idx == 0 and c_idx == 0) else None, zorder=2)

                    if show_points:
                        for i, a_level in enumerate(a_levels):
                            cell_vals = subset[(subset[factor_a_col].astype(str).str.strip() == a_level) &
                                               (subset[factor_b_col].astype(str).str.strip() == b_level)][plot_col]
                            cell_vals = pd.to_numeric(cell_vals, errors='coerce').dropna()
                            if len(cell_vals) > 0:
                                jitter_vals = np.random.normal(0, bar_width * 0.06, len(cell_vals))
                                ax.scatter(x_pos[i] + jitter_vals, cell_vals, color='black', alpha=0.5,
                                           s=12, zorder=10, edgecolors='white', linewidths=0.2)

                ax.set_xticks(x_base)
                ax.set_xticklabels(a_levels, fontsize=7)

                if row_idx == 0:
                    ax.set_title(f'{fc_name} = {c_level}', fontsize=9)
                if c_idx == 0:
                    ax.set_ylabel(f'log10({col})' if log_scale else col, fontsize=8)
                if row_idx == n_analytes - 1:
                    ax.set_xlabel(fa_name, fontsize=8)

            # ANOVA annotation panel (rightmost column)
            if has_results:
                ann_ax = axes[row_idx, n_c]
                ann_ax.axis('off')

                if tw_result is not None:
                    fa_short = fa_name[:12] if len(fa_name) > 12 else fa_name
                    fb_short = fb_name[:12] if len(fb_name) > 12 else fb_name
                    fc_short = fc_name[:12] if len(fc_name) > 12 else fc_name

                    effects = [
                        (fa_short, tw_result.factor_a_pvalue),
                        (fb_short, tw_result.factor_b_pvalue),
                        (fc_short, tw_result.factor_c_pvalue),
                        (f'{fa_short}\u00d7{fb_short}', tw_result.interaction_ab_pvalue),
                        (f'{fa_short}\u00d7{fc_short}', tw_result.interaction_ac_pvalue),
                        (f'{fb_short}\u00d7{fc_short}', tw_result.interaction_bc_pvalue),
                        (f'{fa_short}\u00d7{fb_short}\u00d7{fc_short}', tw_result.interaction_abc_pvalue),
                    ]

                    lines = []
                    for name, p_val in effects:
                        sig = _stars(p_val)
                        p_str = _fmt_p(p_val)
                        lines.append(f'{name}: {p_str} {sig}')

                    ann_text = '\n'.join(lines)
                    any_sig = any(
                        not (p is None or (isinstance(p, float) and math.isnan(p))) and p < 0.05
                        for _, p in effects
                    )
                    bg_color = '#fff3cd' if any_sig else '#f8f9fa'

                    ann_ax.text(0.05, 0.5, ann_text, transform=ann_ax.transAxes,
                                fontsize=6, fontfamily='monospace', va='center', ha='left',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color, alpha=0.9,
                                          edgecolor='#dee2e6'))

        plt.tight_layout()

        # Single legend at top-right, placed after tight_layout so it isn't displaced
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title=fb_name, loc='upper right',
                      fontsize=7, title_fontsize=8, framealpha=0.9,
                      bbox_to_anchor=(0.98, 0.99))

        return fig

    def plot_threeway_interaction_multi_panel(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        factor_a_col: str,
        factor_b_col: str,
        factor_c_col: str,
        threeway_results: Dict = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        factor_a_name: Optional[str] = None,
        factor_b_name: Optional[str] = None,
        factor_c_name: Optional[str] = None,
        ylabel: str = "Concentration",
    ) -> plt.Figure:
        """Multi-panel interaction (line) plots for three-way ANOVA."""
        n_plots = len(value_cols)
        nrows = int(np.ceil(n_plots / ncols))

        if figsize is None:
            figsize = (ncols * 5.5, nrows * 4.5)

        fig = plt.figure(figsize=figsize)
        gs_outer = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.35)

        fa_name = factor_a_name or factor_a_col
        fb_name = factor_b_name or factor_b_col
        fc_name = factor_c_name or factor_c_col

        df = data.copy()
        c_levels = sorted(df[factor_c_col].astype(str).str.strip().unique())
        a_levels = sorted(df[factor_a_col].astype(str).str.strip().unique())
        b_levels = sorted(df[factor_b_col].astype(str).str.strip().unique())
        n_c = len(c_levels)

        palette = self.group_palette[:len(b_levels)]
        x_pos = np.arange(len(a_levels))

        for plot_idx, col in enumerate(value_cols):
            row = plot_idx // ncols
            col_idx = plot_idx % ncols

            if col not in df.columns:
                continue

            tw_result = None
            if threeway_results and col in threeway_results:
                result_obj = threeway_results[col]
                if hasattr(result_obj, 'threeway_result'):
                    tw_result = result_obj.threeway_result
                else:
                    tw_result = result_obj

            gs_inner = gs_outer[row, col_idx].subgridspec(1, n_c, wspace=0.2)

            for c_idx, c_level in enumerate(c_levels):
                ax = fig.add_subplot(gs_inner[0, c_idx])
                subset = df[df[factor_c_col].astype(str).str.strip() == c_level]

                for b_idx, b_level in enumerate(b_levels):
                    means = []
                    sems = []
                    for a_level in a_levels:
                        cell = subset[(subset[factor_a_col].astype(str).str.strip() == a_level) &
                                      (subset[factor_b_col].astype(str).str.strip() == b_level)][col]
                        cell = pd.to_numeric(cell, errors='coerce').dropna()
                        means.append(cell.mean() if len(cell) > 0 else np.nan)
                        sems.append(cell.sem() if len(cell) > 1 else 0)

                    ax.errorbar(x_pos, means, yerr=sems, marker='o', capsize=3,
                               color=palette[b_idx], linewidth=1.5, markersize=5,
                               label=str(b_level) if (plot_idx == 0 and c_idx == 0) else None)

                ax.set_xticks(x_pos)
                ax.set_xticklabels(a_levels, fontsize=7, rotation=0)
                ax.tick_params(labelsize=7)

                if c_idx == 0:
                    ax.set_ylabel(col, fontsize=8)
                ax.set_xlabel(fa_name, fontsize=7)

                if c_idx > 0:
                    ax.set_yticklabels([])

                ax.set_title(f'{fc_name}={c_level}', fontsize=7, pad=3)

        fig.subplots_adjust(hspace=0.55, wspace=0.35)

        # Add shared legend after tight_layout so it stays close to the plots
        handles = []
        labels = []
        for b_idx, b_level in enumerate(b_levels):
            handles.append(plt.Line2D([0], [0], color=palette[b_idx], marker='o', linewidth=1.5))
            labels.append(str(b_level))
        fig.legend(handles, labels, title=fb_name, loc='upper right',
                  fontsize=8, title_fontsize=9, framealpha=0.9,
                  bbox_to_anchor=(0.98, 0.9))

        return fig
    def save_figure(
        self,
        fig: plt.Figure,
        filepath: Union[str, Path],
        formats: List[str] = ['png', 'pdf'],
        dpi: int = 300
    ):
        """Save figure in multiple formats."""
        filepath = Path(filepath)
        
        for fmt in formats:
            save_path = filepath.with_suffix(f'.{fmt}')
            fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight')


def create_summary_figure(
    data: pd.DataFrame,
    sphingolipid_cols: List[str],
    group_col: str,
    totals: pd.DataFrame,
    output_path: Optional[Path] = None,
    units: str = 'ng/mL'
) -> plt.Figure:
    """Create a comprehensive summary figure with multiple panels."""
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Total sphingolipids by group
    ax1 = fig.add_subplot(gs[0, 0])
    combined = pd.concat([data[[group_col]], totals], axis=1)
    if 'total_all' in combined.columns:
        sns.boxplot(data=combined, x=group_col, y='total_all', ax=ax1)
        ax1.set_title('Total Sphingolipids')
        ax1.set_ylabel(f'Concentration ({units})')
    
    # 2. Ceramides vs Sphingomyelins
    ax2 = fig.add_subplot(gs[0, 1])
    if 'total_ceramides' in totals.columns and 'total_sphingomyelins' in totals.columns:
        plot_data = pd.DataFrame({
            'Ceramides': totals['total_ceramides'],
            'Sphingomyelins': totals['total_sphingomyelins'],
            'Group': data[group_col]
        }).melt(id_vars='Group', var_name='Type', value_name='Concentration')
        sns.barplot(data=plot_data, x='Group', y='Concentration', hue='Type', ax=ax2)
        ax2.set_title('Ceramides vs Sphingomyelins')
        ax2.legend(title='')
    
    # 3. Long vs Very Long Chain
    ax3 = fig.add_subplot(gs[0, 2])
    if 'long_chain' in totals.columns and 'very_long_chain' in totals.columns:
        plot_data = pd.DataFrame({
            'Long Chain': totals['long_chain'],
            'Very Long Chain': totals['very_long_chain'],
            'Group': data[group_col]
        }).melt(id_vars='Group', var_name='Type', value_name='Concentration')
        sns.barplot(data=plot_data, x='Group', y='Concentration', hue='Type', ax=ax3)
        ax3.set_title('Long vs Very Long Chain')
        ax3.legend(title='')
    
    # 4. Saturated vs Unsaturated
    ax4 = fig.add_subplot(gs[1, 0])
    if 'saturated' in totals.columns and 'unsaturated' in totals.columns:
        plot_data = pd.DataFrame({
            'Saturated': totals['saturated'],
            'Unsaturated': totals['unsaturated'],
            'Group': data[group_col]
        }).melt(id_vars='Group', var_name='Saturation', value_name='Concentration')
        sns.barplot(data=plot_data, x='Group', y='Concentration', hue='Saturation', ax=ax4)
        ax4.set_title('Saturated vs Unsaturated')
        ax4.legend(title='')
    
    # 5. Composition stacked bar
    ax5 = fig.add_subplot(gs[1, 1:])
    available_cols = [c for c in sphingolipid_cols if c in data.columns][:15]
    if available_cols:
        group_means = data.groupby(group_col)[available_cols].mean()
        row_sums = group_means.sum(axis=1)
        pct_data = group_means.div(row_sums, axis=0) * 100
        pct_data.plot(kind='bar', stacked=True, ax=ax5, 
                     colormap='tab20', edgecolor='white', linewidth=0.3)
        ax5.set_title('Sphingolipid Composition (%)')
        ax5.set_ylabel('Percentage')
        ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Top individual sphingolipids
    ax6 = fig.add_subplot(gs[2, :])
    top_species = data[available_cols].mean().nlargest(8).index.tolist()
    if top_species:
        plot_data = data[[group_col] + top_species].melt(
            id_vars=group_col, var_name='Sphingolipid', value_name='Concentration'
        )
        sns.boxplot(data=plot_data, x='Sphingolipid', y='Concentration', 
                   hue=group_col, ax=ax6)
        ax6.set_title('Top 8 Most Abundant Sphingolipids')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
        ax6.legend(title='Group')
    
    plt.suptitle('Sphingolipid Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
