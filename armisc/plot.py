"""Convenience functions for plotting. It's recommended that you also 
add `plt.tight_layout()` at the end of your plots for consistent sizing.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 72

__all__ = ["fontsize", "labels", "annotate"]


def fontsize(ax, fz=14):
    """Set fontsizes for title, axis labels, and ticklabels.

    Parameters
    ----------
    ax : axis
        matplotlib axis
    fz : int
        font size
    """
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fz)


def labels(
    ax,
    title=None,
    subtitle=None,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    yticklabels=None,
    legend_title=None,
    legend_labels=None,
    legend_loc="upper right",
    fz=14,
    grid=True,
):
    """Assign titles and labels.
    xticklabels and legend_labels should be lists, all others are strings.

    Parameters
    ----------
    ax :
        matplotlib axis
    title :
        figure title
    subtitle :
        figure subtitle
    xlabel :
        x-axis label
    ylabel :
        y-axis label
    xticklabels :
        list of labels for x-axis
    yticklabels :
        list of labels for y-axis
    legend_title :
        legend title
    legend_labels :
        list of legend labels
    legend_loc :
        Legend position: upper right|lower right|upper left|lower left
    fz :
        font size
    grid :
        boolean
    """
    if title is not None:
        if subtitle is None:
            plt.title(title)
        else:
            plt.suptitle(title, y=1, fontsize=fz + 2)
            plt.title(subtitle)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if legend_title is not None and legend_labels is not None:
        handles, ax_legend_labs = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, title=legend_title, loc=legend_loc)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if grid:
        plt.grid(linestyle="dotted")
    fontsize(ax, fz)


def annotate(ax, x, y, text, offset=5):
    """Place text in different locations of the plot.
    x and y are lists of locations on the plot. text can be a list of numbers or strings."""
    for i in range(len(x)):
        ax.text(
            x[i],
            y[i] + offset,
            text[i],
            bbox={"facecolor": "white", "pad": 5},
            horizontalalignment="center",
        )


def panel_label(ax, s: str, x: float = -0.1, y: float = 1.15, fz: float = 16):
    """Label subplot panel for a paper

    Parameters
    ----------
    ax : matplotlib axis
        Axis
    s : str
        Label
    x : float, optional
        x position
    y : float, optional
        yposition
    fz : float, optional
        fontsize
    """
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        fontsize=fz,
        fontweight="bold",
        va="top",
        ha="right",
    )
