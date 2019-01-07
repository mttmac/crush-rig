#!/usr/bin/env python

'''
Define tools to plot crush data.

Written by Matt MacDonald
For CIGITI at the Hospital for Sick Children Toronto
'''


# IMPORTS

from crush_read import *


# PLOT FUNCTIONS

def random(crushes):
    num = np.random.choice(crushes.index)
    print(f"Crush #{num}")
    return crushes.loc[[num], :]


def time_plot(crushes, max_num=10, stress_strain=False, **kwargs):
    """
    Accepts crushes dataframe or subset and plots the transient crush data
    as a time series graph
    """
    options = {'align': True}
    if kwargs:
        options = kwargs

    # Set labels
    if stress_strain:
        labels = ['Strain', 'Stress (MPa)']
    else:
        labels = ['Position (mm)', 'Force (N)']

    # Make plot
    fig = plt.figure()
    p_ax = plt.subplot2grid((2, 7), (0, 0), colspan=5)
    f_ax = plt.subplot2grid((2, 7), (1, 0), colspan=5, sharex=p_ax)
    p_ax.set_ylabel(labels[0])
    f_ax.set_ylabel(labels[1])
    f_ax.set_xlabel('Time (s)')

    gen_plot(crushes, labels[0], max_num=max_num, ax=p_ax, **options)
    gen_plot(crushes, labels[1], max_num=max_num, ax=f_ax, **options)

    names = crushes['Summary'].tolist()[:min(len(crushes), max_num)]
    fig.legend(names, loc='center right',
               prop={'size': 8})


def gen_plot(crushes, labels, max_num=10,
             ax=None, trim=True, align=False, fmt=None):
    """
    Accepts crushes dataframe or subset and plots a single graph
    Input labels must be y label or a tuple of x and y labels (x, y)
    """

    if isinstance(crushes, pd.Series):  # in case a single row is input
        crushes = pd.DataFrame(crushes).T
    if isinstance(labels, str):
        labels = tuple([labels])

    # Prep for aligning data if needed
    lead_time = 1  # seconds, previously: pd.Timedelta('1s')
    if align:
        max_offset = crushes['Data'].apply(crush_duration).max() + lead_time

    # Make plot
    new = ax is None
    if new:
        fig = plt.figure()
        ax = plt.subplot2grid((20, 1), (1, 0), rowspan=19)
        if len(labels) > 1:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        else:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(labels[0])

    for i, num in enumerate(crushes.index):
        if i == max_num:
            break
        crush = crushes.loc[num, 'Data']
        if trim:
            crush = trim_time(crush, lead_time)
        if align:
            crush = rezero_target(crush, max_offset)
        if len(labels) > 1:
            x = crush[labels[0]]
            y = crush[labels[1]]
        else:
            x = crush.index.total_seconds()
            y = crush[labels[0]]
        if fmt is None:
            ax.plot(x, y)
        else:
            ax.plot(x, y, fmt)

    if new:
        names = crushes['Summary'].tolist()[:min(len(crushes), max_num)]
        fig.legend(names, loc='upper center', ncol=2,
                   prop={'size': 8})
    return ax


def stress_plot(crushes, **kwargs):
    """
    Accepts crushes dataframe or subset and plots a stress-strain graph
    """
    gen_plot(crushes, ('Strain', 'Stress (MPa)'), **kwargs)


def stiffness_plot(crushes, **kwargs):
    """
    Accepts crushes dataframe or subset and plots a stiffness-strain graph
    """
    gen_plot(crushes, ('Strain', 'Stiffness (MPa)'), **kwargs)


def fit_plot(crushes, in_time=True, **kwargs):
    """
    Accepts crushes dataframe or subset and plots a stress and the fitted
    polynomial stress curve on teh same graph
    If in_time=False plots the stresses against each other where perfect
    fit would be a linear 1 to 1 relationship
    """
    if in_time:
        ax = gen_plot(crushes, ('Strain', 'Stress (MPa)'), **kwargs)
        gen_plot(crushes, ('Strain', 'Fit Stress (MPa)'),
                 **kwargs, ax=ax, fmt='k:')
    else:
        gen_plot(crushes, ('Stress (MPa)', 'Fit Stress (MPa)'), **kwargs)


def position_plot(crushes, **kwargs):
    """
    Accepts crushes dataframe or subset and plots a position graph
    """
    gen_plot(crushes, 'Position (mm)', **kwargs)


def stage_plot(crushes, labels=None, **kwargs):
    """
    Accepts crushes dataframe or subset and plots a position graph with stages
    """
    if labels is None:
        labels = ['Force (N)']
    if isinstance(labels, str):
        labels = tuple([labels])
    ax = gen_plot(crushes, 'Stage', fmt='k--', **kwargs)
    for label in labels:
        gen_plot(crushes, label, ax=ax, **kwargs)


def force_plot(crushes, raw=False, **kwargs):
    """
    Accepts crushes dataframe or subset and plots a position graph
    """
    if raw:
        gen_plot(crushes, 'Raw Force (N)', **kwargs)
    else:
        gen_plot(crushes, 'Force (N)', **kwargs)


def pressure_plot(crushes, **kwargs):
    """
    Accepts crushes dataframe or subset and plots a position graph
    """
    gen_plot(crushes, 'Pressure (kPa)', **kwargs)
