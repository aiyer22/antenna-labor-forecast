import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

IMPORTANT_TASKS = ["3110", "3180", "3220", "4200"]

color_palette = sns.palettes.color_palette("pastel")
color_palette_2 = sns.palettes.color_palette("muted")
project_colors = {
    "Riptide": color_palette[0],
    "Swordfish Antennas": color_palette[1],
    "Thunderbolt": color_palette[2],
    "Dinosaur": color_palette[3],
    "Hummingbird": color_palette[4],
    "Sailfish": color_palette[5],
}

task_colors = {
    "3110": color_palette[0],
    "3110.1": color_palette_2[0],
    "3110.2": color_palette[5],
    "3110.3": color_palette[6],
    "3110.4": color_palette[7],
    "3180": color_palette[1],
    "3220": color_palette[2],
    "4200": color_palette[3],
    "4200.1": color_palette_2[3],
    "4200.2": color_palette[7],
    "4200.3": color_palette_2[6],
    "4200.4": color_palette_2[7],
}


def plot_value_counts_density(
    grouped_counts,
    xlabel="",
    ylabel="Density",
    title="",
    barlabel_fontsize=7,
    xticklabel_fontsize=9,
    xticklabel_rotation=20,
    barlabel_show_threshold=0.01,
    palette="pastel",
    order=[],
    ax=None,
):
    """
    Plots a bar chart of value counts with density labels.

    Parameters:
    - grouped_counts (pandas.Series): A pandas Series containing the grouped counts.
    - xlabel (str): The label for the x-axis. Default is an empty string.
    - ylabel (str): The label for the y-axis. Default is "Density".
    - title (str): The title of the plot. Default is an empty string.
    - barlabel_fontsize (int): The font size of the bar labels. Default is 7.
    - xticklabel_fontsize (int): The font size of the x-axis tick labels. Default is 9.
    - xticklabel_rotation (int): The rotation angle of the x-axis tick labels. Default is 20.
    - barlabel_show_threshold (float): The threshold for showing the bar labels. Default is 0.01.
    - palette (str): The color palette to use for the bars. Default is "pastel".
    - order (list): The order of the bars. Default is an empty list, which means the order is determined by the index of the grouped_counts.

    Returns:
    - ax (matplotlib.axes.Axes): The matplotlib Axes object containing the plot.
    """

    if not order:
        order = grouped_counts.index.to_list()

    ax = sns.barplot(
        x=grouped_counts.index,
        y=grouped_counts.values,
        order=order,
        palette=palette,
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(
        labels=ax.get_xticklabels(),
        fontsize=xticklabel_fontsize,
        rotation=xticklabel_rotation,
    )

    for bar in ax.containers[0]:
        bar_value = bar.get_height()  # Get the value of the bar
        text = (
            # Don't use scientific notation for small values
            f"{bar_value:.0f}"
            if bar_value >= barlabel_show_threshold
            else f"0"
        )  # Conditional label
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=barlabel_fontsize,
        )

    plt.yticks(fontsize=9)
    plt.title(title)

    return ax


def scan_column(
    df,
    n=5,
    column="expenditure_comment",
    random=True,
    unique_only=True,
    repeats=False,
    remember_path="notebooks\saved_comments.txt",
):
    """
    Scans a column in a DataFrame, samples comments, and provides options to save the comments to a file.

    Args:
        df (pandas.DataFrame): The DataFrame to scan.
        n (int, optional): The number of comments to sample. Defaults to 5.
        column (str, optional): The name of the column to scan. Defaults to "expenditure_comment".
        random (bool, optional): Whether to sample comments randomly. Defaults to True.
        unique_only (bool, optional): Whether to consider only unique comments. Defaults to True.
        repeats (bool, optional): Whether to allow repeated sampling of comments. Defaults to False.
        remember_path (str, optional): The file path to save the comments. Defaults to "notebooks\saved_comments.txt".

    Returns:
        None
    """

    if remember_path:
        assert os.path.exists(remember_path)
        assert remember_path.endswith(".txt")

    df = df.dropna(subset=[column])

    df = df.reset_index(drop=True).dropna(subset=[column])

    if unique_only:
        df = df.drop_duplicates(subset=[column])

    if random:
        sample = df.sample(n)[column]
    else:
        sample = df.head(n)[column]

    # Print the comments
    for comment in sample:
        print(comment)

    if repeats == False:
        df = df.drop(sample.index)

    prompt = input(
        "Press enter to sample more. Enter 's' to save all comments to a file. 'q' or 'esc' to quit:"
    )

    if prompt == "s":
        with open(remember_path, "a") as file:
            for idx, comment in zip(sample.index.to_list(), sample):
                file.write(str(idx) + "\t" + comment + "\n")

    elif prompt == "":
        return scan_column(
            df,
            n=n,
            column=column,
            random=random,
            unique_only=unique_only,
            repeats=repeats,
            remember_path=remember_path,
        )
    elif prompt == "q":
        return


def get_grouped_cumsum(df, col, groupby_col):
    """
    Calculate the cumulative sum of a column in a DataFrame, grouped by another column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    col (str): The name of the column to calculate the cumulative sum.
    groupby_col (str): The name of the column to group by.

    Returns:
    pandas.Series: The cumulative sum of the specified column, grouped by the specified column.
    """
    grouped = df[col].groupby(groupby_col).sum()
    cumulative = np.cumsum(grouped)
    return cumulative


def plot_total_time_by_task_number(df, ax=None):
    total_time_by_task = df.groupby("task_number")["regular_time"].sum()

    ax = plot_value_counts_density(
        total_time_by_task,
        title="Total Time by Task Number",
        xlabel="Task Number",
        ylabel="Labor Time (Hours)",
        xticklabel_rotation=0,
        xticklabel_fontsize=10,
        barlabel_fontsize=9,
        ax=ax,
    )

    return ax


def plot_total_cost_by_task_number(df, ax=None):
    total_cost_by_task = df.groupby("task_number")["raw_cost"].sum()

    ax = plot_value_counts_density(
        total_cost_by_task,
        title="Total Cost by Task Number",
        xlabel="Task Number",
        ylabel="Labor Cost (USD)",
        xticklabel_rotation=0,
        xticklabel_fontsize=10,
        barlabel_fontsize=9,
        ax=ax,
    )

    return ax
