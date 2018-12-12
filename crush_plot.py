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
from pathlib import Path
import glob


PATH = Path('/Users/mattmacdonald/Data/RAWDATA_CRUSH/')


def study_outline(root_folder=None):
    """
    Reads study patients and associated details from a single csv file
    in the root folder containing study outline details, at minimum:
    PAtient Code,Procedure Date,Gender,DOB,Procedure,Tissue,Surgeon,
    Notes,Issues,Histology,Classification
    Assumes all data is kept in sub folders in root folder
    Returns dataframe with Test ID as index
    """
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

    def get_folder_name(row):
        date_fmt = "%Y%m%d"
        fmt = "{date} - {code} - {clsf}"
        date = row['Procedure Date'].strftime(date_fmt)
        code = row['Patient Code'].upper()
        clsf = row['Classification'].upper()
        return fmt.format(**locals())

    study['Folder Name'] = study.apply(get_folder_name, axis=1)

    study.index = study.index + 1  # one indexed
    study = study.rename_axis('Test ID')

    return study


# TODO continue here
def study_data(study):
    """
    Reads all crush data as per study outline dataframe
    Loops over each Patient Code and reference folder and subfolders
    Data csv files must have no titles and a summary line at eof with columns:
    time (seconds), position (um), force (grams)
    Returns dataframe with each crush's data appended as a row
    """
    crushes = DataFrame(columns=['patient', 'protocol', 'load',
                                 'crush', 'summary'])
    for patient in study.index:
        # Assumes folder structure is broken up by study date then patient ID
        path = Path('{}/{}/{}/'.format(study.path[patient],
                                       study.folder[patient],
                                       study.subfolder[patient]))
        data_files = [file for file in os.listdir(path) if file[-4:] == '.csv']
        data_files.sort()

        # Read all patient crush data and add to dataframe
        for filename in data_files:
            data = pd.read_csv(path / filename,
                               names=['time', 'position', 'force'],
                               skipfooter=1, engine='python')  # skip summary
            # Correct units and set index to datetime
            data['position'] = data['position'] / 1000  # um
            data['force'] = (data['force'] / 1000) * 9.81  # grams
            data['time'] = pd.to_timedelta(data.time, unit='s')
            data = data.set_index('time')
            # Units are stored in dataframe as metadata for convenience
            data.units = {'position': 'mm', 'force': 'N'}
            with open(path / filename) as file:
                for i, line in enumerate(file):
                    if i == data.shape[0]:
                        summary = line
                    elif i > data.shape[0]:
                        break
            name_comps = filename.lower().split('_')  # details in csv name

            # Append to end of crushes
            crushes = crushes.append({
                'patient': patient,
                'protocol': name_comps[0],
                'load': name_comps[1].split('.')[0],
                'crush': data,
                'summary': summary
            }, ignore_index=True)

    crushes['protocol'] = crushes.protocol.map({  # fix protocol names
        'constantforce': 'hold',
        'constantpos': 'stop',
        'constantposition': 'stop',
        'multicrush': 'multi-stop'})
    crushes.index.name = 'crush'

    return crushes


def calculate_data(crushes):
    """
    Accepts crushes dataframe, adds calculated data and returns
    """

    # TODO: Add all error checking to top level function

    def rolling_force(crush):
        # Calculate rolling average force
        window = 12  # approximately 1/2 sec
        rolling_force = crush[['force']].rolling(
            window=window, min_periods=int(window / 4), center=True).mean()
        crush['rolling_force'] = rolling_force['force'].fillna(method='bfill')
        crush.units['rolling_force'] = 'N'
        return crush

    def total_time(crush):
        return crush.index[-1] - crush.index[0]

    def sample_period(crush):
        return np.mean(crush.index[1:] - crush.index[:-1])

    def sample_rate(crush):
        return 1E9 / float(sample_period(crush))

    def touch_time(crush):
        # TODO: fix, this is error prone for noisy crush transients
        touch_metric = 0.1  # Assume 0.1 N increase equals touch
        if 'rolling_force' in crush.columns:
            force = crush['rolling_force']
        else:
            force = crush['force']
        return (force - force[0] >= touch_metric).argmax()

    def hanging_load(crush):
        touch = touch_time(crush)
        return crush.force[crush.index <= touch].mean()

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
