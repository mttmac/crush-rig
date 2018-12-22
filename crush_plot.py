#!/usr/bin/env

'''
Define tools to use in a jupyter notebook for analyzing crush data.

Written by Matt MacDonald
For CIGITI at the Hospital for Sick Children Toronto
'''

import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns

import os
import platform
from pathlib import Path
import glob
import re

from pdb import set_trace

# CONSTANTS
PATH = Path('/Users/mattmacdonald/Data/RAWDATA_CRUSH/')
PIN_DIAM = 5.0  # mm


def study_outline(root_folder=None):
    """
    Reads study patients and associated details from a single csv file
    in the root folder containing study outline details, at minimum:
    Patient Code,Procedure Date,Gender,DOB,Procedure,Tissue,Surgeon,
    Notes,Issues,Histology,Classification
    Assumes all data is kept in sub folders in root folder
    Returns dataframe with Test ID as index
    """

    def get_folder_name(row):
        date_fmt = "%Y%m%d"
        fmt = "{date} - {code} - {clsf}"
        date = row['Procedure Date'].strftime(date_fmt)
        code = row['Patient Code'].upper()
        clsf = row['Classification'].upper()
        return fmt.format(**locals())

    # Read outline file
    if root_folder is None:
        root_folder = Path.cwd()
    files = glob.glob(str(root_folder / '*.csv'))
    assert len(files) == 1, "Root data folder must only contain one .csv file."
    study = pd.read_csv(root_folder / files[0])

    # Cleanup and organize information, including data subfolders
    study = study.fillna('N/A')
    study['Procedure Date'] = pd.to_datetime(study['Procedure Date'],
                                             format='%Y-%m-%d')
    study['DOB'] = pd.to_datetime(study['DOB'],
                                  format='%m/%d/%Y')
    study['Age'] = study['Procedure Date'] - study['DOB']
    study['Folder Name'] = study.apply(get_folder_name, axis=1)

    study.index = study.index + 1  # one indexed
    study = study.rename_axis('Test ID')

    return study


def study_data(study):
    """
    Reads all crush data as per study outline dataframe
    Loops over each Test ID and reads subfolder
    Data csv files must be unchanged from the output from crush.py
    Returns dataframe with each crush as a separate row
    """

    def get_creation_date(filepath):
        if platform.system() == 'Windows':
            epoch_time = os.path.getctime(filepath)
        else:
            # Does not support LINUX!
            epoch_time = os.stat(filepath).st_birthtime
        return pd.to_datetime(epoch_time, unit='s')

    features = ['Test ID',
                'Datetime',
                'Patient',
                'Tissue',
                'Protocol',
                'Load',
                'Summary',
                'Data']
    crushes = pd.DataFrame(columns=features)
    crush_pattern = re.compile(r"(?P<protocol>\w+)-"
                               r"(?P<load>\d+.?\d*)g"
                               r"-?\d*.csv")
    for test in study.index:
        path = PATH / study.loc[test, 'Folder Name']
        files = [path / file for file in os.listdir(path)]

        # Read all patient crush data and add to dataframe
        for file in files:
            crush_match = crush_pattern.match(file.name)
            if not crush_match:
                continue

            # Read and set index to timestamp
            data = pd.read_csv(file)
            data['Timestamp (s)'] = pd.to_timedelta(data['Timestamp (s)'],
                                                    unit='s')
            data = data.set_index('Timestamp (s)')

            # Parse meta data and append to end of crushes
            crush_dict = {
                'Test ID': test,
                'Datetime': get_creation_date(file),
                'Patient': study.loc[test, 'Patient Code'].upper(),
                'Tissue': study.loc[test, 'Classification'].upper(),
                'Protocol': crush_match.group('protocol').lower(),
                'Load': f"{float(crush_match.group('load')):.0f}g",
                'Data': data}
            crush_dict['Summary'] = "Patient {} ({}), {} crush at {}".format(
                                    crush_dict['Patient'],
                                    crush_dict['Tissue'],
                                    crush_dict['Protocol'],
                                    crush_dict['Load'])
            crushes = crushes.append(crush_dict, ignore_index=True)

    crushes.index.name = 'Crush'
    return crushes


def sample_period(crush):
    return pd.Timedelta(np.mean(crush.index[1:] - crush.index[:-1]))


def sample_rate(crush):
    # Returns the average sample rate in Hz
    return 1 / sample_period(crush).total_seconds()


def total_time(crush):
    return crush.index[-1]


def stage_times(crush):
    # Return time of transition for each stage
    # 0 for approach, 1 for crush, 2 for target, 3 for release
    times = [pd.Timedelta(0)]
    for stage in range(1, 4):
        times.append((crush['Stage'] == stage).argmax())
    return tuple(times)


def stage_durations(crush):
    times = [*stage_times(crush), total_time(crush)]
    durations = []
    for transitions in zip(times[1:], times[:-1]):
        durations.append(transitions[0] - transitions[1])
    return tuple(durations)


def stage_repetition(crush):
    # Returns the start of stage 0 again if any
    after = release_time(crush)
    rep = (crush.loc[after:, 'Stage'] == 0).argmax()
    if rep == after:
        return None
    return rep


def contact_time(crush):
    return stage_times(crush)[1]


def contact_duration(crush):
    durations = stage_durations(crush)
    return durations[1] + durations[2]


def contact_position(crush):
    return crush.loc[contact_time(crush), 'Position (mm)']


def contact_force(crush):
    return crush.loc[contact_time(crush), 'Force (N)']


def approach_duration(crush):
    return stage_durations(crush)[0]


def movement_duration(crush):
    durations = stage_durations(crush)
    return durations[0] + durations[1]


def crush_duration(crush):
    return stage_durations(crush)[1]


def target_time(crush):
    return stage_times(crush)[2]


def target_duration(crush):
    return stage_durations(crush)[2]


def target_position(crush):
    return crush.loc[target_time(crush), 'Position (mm)']


def target_force(crush):
    return crush.loc[target_time(crush), 'Force (N)']


def release_time(crush):
    return stage_times(crush)[3]


def release_duration(crush):
    return stage_durations(crush)[3]


def release_position(crush):
    return crush.loc[release_time(crush), 'Position (mm)']


def release_force(crush):
    return crush.loc[release_time(crush), 'Force (N)']


def crush_distance(crush):
    return target_position(crush) - contact_position(crush)


def force_relaxation(crush):
    return target_force(crush) - release_force(crush)


# TODO refine this definition to be zero just before contact
def hanging_force(crush):
    return crush['Force (N)'][crush.index <= contact_time(crush)].mean()


def smooth_force(crush):
    # Calculate force with a low pass butterworth filter
    # Intent is to smooth out noisy force sensor readings
    # Raw readings stored for future reference

    force = crush['Force (N)']
    if 'Raw Force (N)' not in crush.columns:
        crush['Raw Force (N)'] = force.copy()

    # Split before and after the release stage to avoid artifacts
    rel = release_time(crush)
    pre = force.index < rel
    post = force.index >= rel
    forces = [force[pre], force[post]]

    # Filter on force data
    N = 3  # Filter order
    Wn = 0.1  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    for force in forces:
        force = signal.filtfilt(B, A, force.values)
    return crush


def add_stress(crush):
    # Calculate stress
    pin_area = np.pi * (PIN_DIAM / 2) ** 2
    crush['Stress (MPa)'] = crush['Force (N)'] / pin_area
    no_contact_mask = (crush['Stage'] == 0) | (crush['Stage'] == 3)
    crush.loc[no_contact_mask, 'Stress (MPa)'] = 0
    return crush


def add_strain(crush):
    # Calculate strain (compressive)
    thickness = abs(contact_position(crush))
    abs_pos = crush['Position (mm)'].abs()
    strain = (thickness - abs_pos) / thickness
    strain[strain < 0] = 0
    crush['Strain'] = strain
    return crush


def add_stiffness(crush, order=3, exponential=True):
    """
    Fits a polynomial curve to stress vs strain to estimate strain-dependent
    stiffness, 3rd order by default, fits to log stress by default
    Only calculates crush stage with NaNs elsewhere
    """
    crush['Fit Stress (Mpa)'] = np.nan
    crush['Stiffness (Mpa)'] = np.nan

    mask = crush['Stage'] == 1  # crush
    x = crush.loc[mask, 'Strain']
    y = crush.loc[mask, 'Stress (MPa)']

    if exponential:
        y = np.log(y)

    z = np.polyfit(x, y, order)
    f = np.poly1d(z)
    df = f.deriv(1)
    y_h = f(x)
    dy_h = df(x)

    if exponential:
        y_h = np.exp(y_h)
        dy_h = y_h * dy_h

    crush.loc[mask, 'Fit Stress (MPa)'] = y_h
    crush.loc[mask, 'Stiffness (MPa)'] = dy_h
    return crush


def tare_force(crush):
    """
    Accepts crush dataframe, shifts to account for hanging load and returns
    """
    tare = hanging_force(crush)
    assert abs(tare) < 0.25, f"Excessive hanging force detected: {tare:.3f}"
    crush['Force (N)'] = crush['Force (N)'] - tare
    return crush


def trim_time(crush, lead_time):
    """
    Accepts crush dataframe, trims N sec before contact and after release
    Rezeros the index to the start
    lead_time must be pd.Timedelta object
    """
    crush = crush[crush.index >= (contact_time(crush) - lead_time)]
    crush = crush[crush.index < release_time(crush)]
    crush.index = crush.index - crush.index[0]
    return crush


def rezero_target(crush, offset):
    """
    Rezeros index to an optional offset to the target
    offset must be pd.Timedelta object
    """
    time = target_time(crush)
    crush.index = crush.index - (time - offset)
    return crush


def select_stage(crush, stage):
    # 0 for approach, 1 for crush, 2 for target, 3 for release
    stages = {'approach': 0,
              'crush': 1,
              'target': 2,
              'release': 3}
    if stage in stages.keys():
        stage = stages[stage]
    assert stage in crush['Stage'].values, "Stage input not found in transient"
    return crush.loc[crush['Stage'] == stage, :]


def split(crushes):
    """
    Divides all multi-crushes into seperate crush instances
    Tracks repetitions with a new repetition column
    Suggest running before modify() or calculate()
    """
    crushes['Repetition'] = 0
    multis = crushes[crushes['Protocol'].str.contains('multi')]

    debug = False  # TODO remove

    for num in multis.index:
        crush = multis.loc[num, 'Data']
        summary = multis.loc[num, 'Summary'].split('crush')
        crush_dict = {
            'Test ID': multis.loc[num, 'Test ID'],
            'Datetime': multis.loc[num, 'Datetime'],
            'Patient': multis.loc[num, 'Patient'],
            'Tissue': multis.loc[num, 'Tissue'],
            'Protocol': multis.loc[num, 'Protocol'][6:],  # remove multi_
            'Load': multis.loc[num, 'Load']}

        rep_num = 0
        searching = True
        while searching:

            if debug:
                set_trace()

            rep = stage_repetition(crush)
            if rep is None:
                sub_crush = crush.copy()
                searching = False
            else:
                sub_crush = crush.loc[crush.index < rep, :].copy()
                crush = crush.loc[rep:, :]

            crush_dict.update({
                'Summary': f"{summary[0]}crush [{rep_num}]{summary[1]}",
                'Data': sub_crush,
                'Repetition': rep_num})
            crushes = crushes.append(crush_dict, ignore_index=True)
            rep_num += 1

        debug = False

    for num in multis.index.tolist():
        crushes = crushes.drop(num)

    return crushes


def modify(crushes):
    """
    Accepts crushes dataframe, modifies transient data and returns
    """
    crushes['Data'] = crushes['Data'].apply(tare_force)
    crushes['Data'] = crushes['Data'].apply(smooth_force)
    crushes['Data'] = crushes['Data'].apply(add_stress)
    crushes['Data'] = crushes['Data'].apply(add_strain)
    crushes['Data'] = crushes['Data'].apply(add_stiffness)
    return crushes


def calculate(crushes):
    """
    Adds calculated statistics about each crush transient and returns
    Suggest running modify() first to get expected results
    """

    def to_force(weight):
        """
        Converts weight in grams to force in N at standard earth gravity.
        """
        return 9.81 * weight / 1000

    def to_stress(force):
        pin_area = np.pi * (PIN_DIAM / 2) ** 2
        return force / pin_area

    def to_strain(delta, length):
        delta, length = abs(delta), abs(length)
        return (length - delta) / length  # compressive positive

    for i, num in enumerate(crushes.index):
        crush = crushes.loc[num, 'Data']

        # Target force of crush
        load_g = float(crushes.loc[num, 'Load'][:-1])
        crushes.loc[num, 'Target (N)'] = to_force(load_g)

        # Stress relaxation after target achieved
        crushes.loc[num, 'Relaxation (MPa)'] = to_stress(
            force_relaxation(crush))

    return crushes


def random(crushes):
    num = np.random.choice(crushes.index)
    print(f"Crush #{num}")
    return crushes.loc[[num], :]


def time_plot(crushes, max_num=10, **kwargs):
    """
    Accepts crushes dataframe or subset and plots the transient crush data
    as a time series graph
    """
    options = {'align': True}
    if kwargs:
        options = kwargs

    # Make plot
    fig = plt.figure()
    p_ax = plt.subplot2grid((2, 7), (0, 0), colspan=5)
    f_ax = plt.subplot2grid((2, 7), (1, 0), colspan=5, sharex=p_ax)
    p_ax.set_ylabel('Position (mm)')
    f_ax.set_ylabel('Force (N)')
    f_ax.set_xlabel('Time (s)')

    gen_plot(crushes, 'Position (mm)', max_num=max_num, ax=p_ax, **options)
    gen_plot(crushes, 'Force (N)', max_num=max_num, ax=f_ax, **options)

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
    lead_time = pd.Timedelta('1s')
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
    Accepts crushes dataframe or subset and plots a stiffness vs strain graph
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


if __name__ == "__main__":
    study = study_outline(PATH)
    crushes = study_data(study)
    crushes = split(crushes)
    crushes = modify(crushes)
    crushes = calculate(crushes)
    # time_plot(crushes)
    # position_plot(crushes)
    # force_plot(crushes)
    # stress_plot(crushes)
    # stiffness_plot(crushes)
    # plt.show()
