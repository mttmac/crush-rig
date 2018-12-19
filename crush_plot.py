#!/usr/bin/env

'''
Define tools to use in a jupyter notebook for analyzing crush data.

Written by Matt MacDonald
For CIGITI at the Hospital for Sick Children Toronto
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import platform
from pathlib import Path
import glob
import re

from pdb import set_trace


PATH = Path('/Users/mattmacdonald/Data/RAWDATA_CRUSH/')


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

    def to_force(weight):
        """
        Converts weight in grams to force in N at standard earth gravity.
        """
        return 9.81 * weight / 1000

    crushes = pd.DataFrame(columns=['Test ID',
                                    'Datetime',
                                    'Patient',
                                    'Tissue',
                                    'Protocol',
                                    'Load',
                                    'Target',
                                    'Data',
                                    'Summary'])
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
                'Patient': study.loc[test, 'Patient Code'],
                'Protocol': crush_match.group('protocol'),
                'Tissue': study.loc[test, 'Classification'],
                'Load': f"{float(crush_match.group('load')):.0f}g",
                'Target': to_force(float(crush_match.group('load'))),
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


def contact_time(crush):
    return stage_times(crush)[1]


def contact_duration(crush):
    durations = stage_durations(crush)
    return durations[1] + durations[2]


def contact_position(crush):
    return crush.loc[contact_time(crush), 'Position (mm)']


def target_time(crush):
    return stage_times(crush)[2]


def target_duration(crush):
    return stage_durations(crush)[2]


def target_position(crush):
    return crush.loc[target_time(crush), 'Position (mm)']


def crush_distance(crush):
    return target_position(crush) - contact_position(crush)


# TODO refine this definition to be zero just before contact
def hanging_force(crush):
    return crush['Force (N)'][crush.index <= contact_time(crush)].mean()


def release_time(crush):
    return stage_times(crush)[3]


def smooth_force(crush, window=None):
    # Calculate rolling average force with half second averaging window
    # Intent is to smooth out noisy force sensor readings
    if window is None:
        window = max(10, int(sample_rate(crush) / 2))
    crush = crush.copy()
    crush['Force (N)'] = crush['Force (N)'].rolling(
        window=window,
        min_periods=int(window / 4),
        center=True).mean().fillna(method='bfill')
    return crush


def add_stress(crush):
    # Calculate stress
    pin_diam = 5.0  # mm
    pin_area = np.pi * (pin_diam / 2) ** 2
    crush['Stress (MPa)'] = crush['Force (N)'] / pin_area
    return crush


def add_strain(crush):
    # Calculate strain (compressive)
    thickness = abs(contact_position(crush))
    abs_pos = crush['Position (mm)'].abs()
    strain = (thickness - abs_pos) / thickness
    strain[strain < 0] = 0
    crush['Strain'] = strain
    return crush


def tare(crush):
    """
    Accepts crush dataframe, shifts to account for hanging load and returns
    """
    tare = hanging_force(crush)
    assert abs(tare) < 0.25, f"Excessive hanging force detected: {tare:.3f}"
    crush['Force (N)'] = crush['Force (N)'] - tare
    return crush


def trim(crush):
    """
    Accepts crush dataframe, trims N sec before contact and after release
    Index is rezeroed
    """
    lead_sec = 1
    lead_time = pd.Timedelta(f'{int(lead_sec)}s')
    crush = crush[crush.index >= (contact_time(crush) - lead_time)]
    crush = crush[crush.index <= release_time(crush)]
    crush.index = crush.index - crush.index[0]
    return crush


def modify(crushes):
    """
    Accepts crushes dataframe, modifies transient data and returns
    """
    crushes['Data'] = crushes['Data'].apply(smooth_force)
    crushes['Data'] = crushes['Data'].apply(tare)
    crushes['Data'] = crushes['Data'].apply(add_stress)
    crushes['Data'] = crushes['Data'].apply(add_strain)
    crushes['Data'] = crushes['Data'].apply(trim)
    return crushes


def time_plot(crushes, max_num=8):
    """
    Accepts crushes dataframe or subset and plots the transient crush data
    as a time series graph
    """

    if isinstance(crushes, pd.Series):  # in case a single row is input
        crushes = pd.DataFrame(crushes).T

    # Strings shorthand
    time = 'Time (s)'
    pos = 'Position (mm)'
    force = 'Force (N)'

    # Make plot
    fig, axes = plt.subplots(nrows=2, sharex=True)
    for i, num in enumerate(crushes.index):
        if i == max_num:
            break
        crush = crushes.loc[num, 'Data'].copy()
        crush.index = crush.index.total_seconds()
        crush.plot(y=pos, ax=axes[0], legend=False)
        crush.plot(y=force, ax=axes[1], legend=False)

    names = crushes['Summary'].tolist()
    axes[0].legend(names, loc='best')
    axes[1].set_xlabel(time)
    axes[0].set_ylabel(pos)
    axes[1].set_ylabel(force)


def stress_plot(crushes, max_num=8):
    """
    Accepts crushes dataframe or subset and plots a stress-strain graph
    """

    if isinstance(crushes, pd.Series):  # in case a single row is input
        crushes = pd.DataFrame(crushes).T

    # Strings shorthand
    stress = 'Stress (MPa)'
    strain = 'Strain'

    # Make plot
    plt.figure()
    ax = plt.gca()
    for i, num in enumerate(crushes.index):
        if i == max_num:
            break
        crush = crushes.loc[num, 'Data'].copy()
        crush.index = crush.index.total_seconds()
        plt.plot(crush[strain], crush[stress])

    names = crushes['Summary'].tolist()
    ax.legend(names, loc='best')
    ax.set_xlabel(strain)
    ax.set_ylabel(stress)


if __name__ == "__main__":
    study = study_outline(PATH)
    crushes = study_data(study)
    crushes = modify(crushes)
    time_plot(crushes)
    stress_plot(crushes)
    plt.show()
