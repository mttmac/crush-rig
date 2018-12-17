#!/usr/bin/env

'''
Define tools to use in a jupyter notebook for analyzing crush data.

Written by Matt MacDonald
For CIGITI at the Hospital for Sick Children Toronto
'''

from pandas import Series, DataFrame
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

    crushes = DataFrame(columns=['Test ID',
                                 'Datetime',
                                 'Protocol',
                                 'Load',
                                 'Target',
                                 'Data'])
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
            crushes = crushes.append({
                'Test ID': test,
                'Datetime': get_creation_date(file),
                'Protocol': crush_match.group('protocol'),
                'Load': f"{float(crush_match.group('load')):.0f}g",
                'Target': to_force(float(crush_match.group('load'))),
                'Data': data}, ignore_index=True)

    crushes.index.name = ' Crush'
    return crushes


def calculate_data(crushes):
    """
    Accepts crushes dataframe, adds calculated data and returns
    """

    # TODO: Add all error checking to top level function

    def sample_period(crush):
        return pd.Timedelta(np.mean(crush.index[1:] - crush.index[:-1]))

    def sample_rate(crush):
        # Returns the average sample rate in Hz
        return 1 / sample_period(crush).total_seconds()

    def total_time(crush):
        return pd.Timedelta(crush.index[-1] - crush.index[0])

    def stage_times(crush):
        # Return time of transition for each stage
        # 0 for approach, 1 for crush, 2 for target, 3 for release
        times = []
        for stage in range(4):
            times.append((crush['Stage'] == stage).argmax())
        return tuple(times)

    def contact_time(crush):
        return stage_times(crush)[1]

    def hanging_load(crush):
        return crush['Force (N)'][crush.index <= contact_time(crush)].mean()

    def use_rolling_force(crush):
        # Calculate rolling average force with half second averaging window
        # Intent is to smooth out noisy force sensor readings
        window = max(10, int(sample_rate(crush) / 2))
        mod_crush = crush.copy()
        mod_crush['Force (N)'] = crush['Force (N)'].rolling(
            window=window,
            min_periods=int(window / 4),
            center=True).mean().fillna(method='bfill')
        return mod_crush

    def align_contact(crushes):
        """
        Accepts crushes dataframe, adds aligned data and returns
        """

        aligned_crushes = []
        for i in crushes.index:
            crush = crushes.loc[i, 'crush'].copy()
            touch_time = crushes.loc[i, 'touch_time']
            touch_position = crush.position[touch_time]
            tare_force = crushes.loc[i, 'hanging_load']
            origin_time = crush.index[0]

            # Offset transient data
            crush['position'] = crush.position - touch_position
            crush['force'] = crush.force - tare_force
            crush['rolling_force'] = crush.rolling_force - tare_force

            # Trim leading data
            crush = crush[crush.index >= (touch_time - pd.Timedelta('1s'))]
            # Rezero index
            crush.index = crush.index - (crush.index[0] - origin_time)

            aligned_crushes.append(crush)

        crushes['aligned_crush'] = aligned_crushes
        return crushes

    # Adds to crush transient data
    crushes['crush'] = crushes.crush.apply(rolling_force)

    # Adds to crush data
    crushes['total_time'] = crushes.crush.apply(total_time)
    crushes['sample_period'] = crushes.crush.apply(sample_period)
    crushes['sample_rate'] = crushes.crush.apply(sample_rate)
    crushes['touch_time'] = crushes.crush.apply(touch_time)
    crushes['hanging_load'] = crushes.crush.apply(hanging_load)
    crushes.units = {'sample_rate': 'Hz',
                     'hanging_load': 'N'
                     }  # only if not obvious from dtype

    # Adds crush transient subsets
    crushes = align_contact(crushes)

    return crushes


def plot_data(crushes, align=False):
    """
    Accepts crushes dataframe or subset and plots the transient crush data
    """

    if isinstance(crushes, Series):  # in case a single row is input
        crushes = DataFrame(crushes).T

    fig, axes = plt.subplots(nrows=2, sharex=True)
    max_ncrush = 10
    for i in crushes.index:
        if i == max_ncrush:
            break
        if align:
            data = crushes.loc[i, 'aligned_crush']
        else:
            data = crushes.loc[i, 'crush']
        data.plot(y='position', ax=axes[0], legend=False)
        data.plot(y='rolling_force', ax=axes[1], legend=False)
    names = ('Patient ' + crushes.patient.map(str) + ': ' + crushes.load +
             ' ' + crushes.protocol + ' crush').tolist()
    axes[0].legend(names, loc='best')
    axes[0].set_ylabel('position (mm)')
    axes[1].set_ylabel('force (N)')
    plt.show()


if __name__ == "__main__":
    study = study_outline('study.csv', 'study-data')
    crushes = study_data(study)
    crushes = calculate_data(crushes)
    plot_data(crushes, align=True)
