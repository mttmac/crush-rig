#!/usr/bin/env python

'''
Define tools to read and analyze crush data.

Written by Matt MacDonald
For CIGITI at the Hospital for Sick Children Toronto
'''


# IMPORTS

import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import os
import platform
from pathlib import Path
import glob
import re

from pdb import set_trace


# CONSTANTS

PATH = Path('/Users/mattmacdonald/Data/RAWDATA_CRUSH_PAPER2/')
PIN_DIAM = 5.0  # mm


# IMPORT FUNCTIONS

def study_outline(root_folder=None):
    """
    Reads study patients and associated details from a single csv file
    in the root folder containing study outline details, at minimum:
    Patient Code,Procedure Date,Gender,DOB,Procedure,Tissue,Surgeon,
    Notes,Issues,Histology,Classification
    File must be named "*MASTERLIST.csv"
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
    files = glob.glob(str(root_folder / '*MASTERLIST.csv'))
    assert len(files) == 1, ('Root data folder must contain one master '
                             'csv file.')
    study = pd.read_csv(root_folder / files[0])

    # Cleanup and organize information, including data subfolders
    study = study.fillna('N/A')
    study['Procedure Date'] = pd.to_datetime(study['Procedure Date'],
                                             format='%m/%d/%Y')
    study['DOB'] = pd.to_datetime(study['DOB'],
                                  format='%m/%d/%Y')
    study['Age'] = study['Procedure Date'] - study['DOB']
    study['Folder Name'] = study.apply(get_folder_name, axis=1)

    study.index = study.index + 1  # one indexed
    study = study.rename_axis('Test ID')

    return study


def study_targets(root_folder=None):
    """
    Reads targets for the dataset from a csv file in the root folder
    File must be named "*TARGETS.csv"
    Returns dataframe
    """
    # Read targets file
    if root_folder is None:
        root_folder = Path.cwd()
    files = glob.glob(str(root_folder / '*TARGETS.csv'))
    assert len(files) == 1, ('Root data folder must contain one targets '
                             'csv file.')
    targets = pd.read_csv(root_folder / files[0])

    # Cleanup and remove invalid targets
    path_scores = []
    path_scores.append(pd.to_numeric(targets['Corwyn Score']
                                     .replace('W', np.nan)))
    path_scores.append(pd.to_numeric(targets['Cathy Score']
                                     .replace('W', np.nan)))
    valid = ~(path_scores[0].isna() & path_scores[1].isna())
    has_both = ~(path_scores[0].isna() | path_scores[1].isna())
    assert not np.any(has_both), "Implement averaging of pathological scores"

    # Use whichever pathologist score is available without averaging
    path_score = pd.concat([path_scores[0][valid], path_scores[1][valid]],
                           axis=1).max(axis=1)
    path_score.name = 'Trauma Score'  # 0, 1, 2, 3 categorical

    # Get pathologist names
    paths = pd.Series(index=path_scores[0].index,
                      name='Pathologist')
    for i in range(len(path_scores)):
        scores = path_scores[i]
        name = scores.name.split()[0]
        scores = ~scores.isna()
        paths[scores] = name
    paths = paths[valid]

    id_labels = ['Patient Code', 'Protocol', 'Tissue', 'Load (g)']
    abs_deltas = targets.loc[valid, 'Absolute Delta (um)']
    deltas = targets.loc[valid, 'Percent Delta']
    pscores = targets.loc[valid, 'P Score']
    ids = targets.loc[valid, id_labels]
    for label in id_labels:
        if label == 'Load (g)':
            func = int
        else:
            func = str.upper
        ids[label] = ids[label].apply(func)
    targets = pd.concat([ids,
                         abs_deltas,
                         deltas,
                         pscores,
                         paths,
                         path_score], axis=1)

    return targets


def study_data(study):
    """
    Reads all crush data as per study outline dataframe
    Loops over each Test ID and reads subfolder
    Data csv files must be unchanged from the output from crush.py
    Returns dataframe with each crush as a separate row
    """

    features = ['Test ID',
                'Patient',
                'Protocol',
                'Tissue',
                'Gender',
                'Age (days)',
                'Load (g)',
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
                'Patient': study.loc[test, 'Patient Code'].upper(),
                'Protocol': crush_match.group('protocol').upper(),
                'Tissue': study.loc[test, 'Classification'].upper(),
                'Gender': study.loc[test, 'Gender'].upper(),
                'Age (days)': int((study.loc[test, 'Procedure Date'] -
                                   study.loc[test, 'DOB']).days),
                'Load (g)': int(float(crush_match.group('load'))),
                'Data': data}
            crush_dict['Summary'] = "Patient {} ({}), {} crush at {}g".format(
                                    crush_dict['Patient'],
                                    crush_dict['Tissue'],
                                    crush_dict['Protocol'],
                                    crush_dict['Load (g)'])
            crushes = crushes.append(crush_dict, ignore_index=True)

    types = {'Age (days)': np.int64,
             'Load (g)': np.int64}
    crushes = crushes.astype(types)
    crushes.index.name = 'Crush'
    return crushes


# ANALYSIS FUNCTIONS

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
        times.append((crush['Stage'] == stage).idxmax())
    return tuple(times)


def stage_durations(crush):
    times = [*stage_times(crush), total_time(crush)]
    durations = []
    for transitions in zip(times[1:], times[:-1]):
        delta = (transitions[0] - transitions[1]).total_seconds()
        durations.append(delta)
    return tuple(durations)


def stage_repetition(crush):
    # Returns the start of stage 0 again if any
    after = release_time(crush)
    rep = (crush.loc[after:, 'Stage'] == 0).idxmax()
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


def target_relaxation(crush):
    return target_force(crush) - release_force(crush)


def target_movement(crush):
    return release_position(crush) - target_position(crush)


def target_error(crush, load):

    def to_force(weight):
        return 9.81 * weight / 1000

    if isinstance(load, str) and (load[-1] == 'g'):
        load = load[:-1]
    set_force = to_force(float(load))
    return target_force(crush) - set_force


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


# TODO refine this definition to be zero just before contact
def hanging_force(crush):
    return crush['Force (N)'][crush.index < contact_time(crush)].mean()


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

    # Filter on force data
    N = 3  # Filter order
    Wn = 0.2  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    crush.loc[pre, 'Force (N)'] = signal.filtfilt(B, A, force[pre].values)
    crush.loc[post, 'Force (N)'] = signal.filtfilt(B, A, force[post].values)
    return crush


def add_pressure(crush):
    # Calculate pressure applied (same as stress)
    pin_area = np.pi * (PIN_DIAM / 2) ** 2
    crush['Pressure (kPa)'] = 1000 * crush['Force (N)'] / pin_area
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


def add_stiffness(crush, n_pieces=10):
    # Calculate the stiffness based on the piecewise slope of the stress
    # strain curve using the range of sample points before each time point.
    # The range is calculated based on the number of sample points divided
    # up into the number of pieces

    def slope(x, y):
        x = np.stack([x, np.ones(len(x))], axis=1)
        line = np.linalg.lstsq(x, y, rcond=None)
        return line[0][0]

    crush['Stiffness (MPa)'] = np.nan
    mask = crush['Stage'] == 1  # crush
    strain = crush.loc[mask, 'Strain'].values
    stress = crush.loc[mask, 'Stress (MPa)'].values
    N = len(strain)
    if N // n_pieces < 1:
        return crush  # no valid stiffness
    indices = range(0, N, N // n_pieces)
    index_ranges = [(i[1], slice(i[0], i[1], 1))
                    for i in zip(indices[:-1], indices[1:])]

    stiff = np.ones(N) * np.nan
    for idx, rng in index_ranges:
        stiffness = slope(strain[rng], stress[rng])
        if stiffness < 0:
            continue  # leave as nan
        stiff[idx] = stiffness

    crush.loc[mask, 'Stiffness (MPa)'] = stiff
    return crush


def add_stiffness_fit(crush, order=3, exponential=True, percentiles=False):
    """
    Fits a polynomial curve to stress vs strain to estimate strain-dependent
    stiffness, 3rd order by default, fits to log stress by default
    Only calculates crush stage with NaNs elsewhere
    Optionally can return calculated values at percentiles of strain
    """
    crush['Fit Stress (MPa)'] = np.nan
    crush['Stiffness (MPa)'] = np.nan

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

    if percentiles:
        percent_x = [x * 0.1 for x in range(0, 11)]
        percent_y = f(percent_x)
        percent_dy = df(percent_x)

    if exponential:
        y_h = np.exp(y_h)
        dy_h = y_h * dy_h

        if percentiles:
            percent_y = np.exp(percent_y)
            percent_dy = percent_y * percent_dy

    crush.loc[mask, 'Fit Stress (MPa)'] = y_h
    crush.loc[mask, 'Stiffness (MPa)'] = dy_h

    if percentiles:
        return crush, zip(percent_x, percent_y, percent_dy)

    return crush


def tare_force(crush):
    """
    Accepts crush dataframe, shifts to account for hanging load and returns
    """
    tare = hanging_force(crush)
    if abs(tare) >= 0.25:
        set_trace()
    # assert abs(tare) < 0.25, f"Excessive hanging force detected: {tare:.3f}"
    crush['Force (N)'] = crush['Force (N)'] - tare
    return crush


def trim_time(crush, lead_time):
    """
    Accepts crush dataframe, trims N sec before contact and after release
    """
    lead_time = pd.Timedelta(lead_time)
    crush = crush[crush.index >= (contact_time(crush) - lead_time)]
    crush = crush[crush.index < release_time(crush)]
    return crush


def rezero(crush, offset=0, zero_index=None):
    """
    Rezeros the index with an optional offset from zero
    Can optionally specify an index to be zero other than the first
    """
    if zero_index is None:
        zero_index = crush.index[0]
    offset = pd.Timedelta(offset)
    crush.index = crush.index - (zero_index - offset)
    return crush


def rezero_target(crush, offset=0):
    """
    Rezeros index to an optional offset to the target time
    """
    return rezero(crush, offset, zero_index=target_time(crush))


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


def modify(crushes):
    """
    Accepts crushes dataframe, modifies transient data and returns
    """
    crushes['Data'] = crushes['Data'].apply(tare_force)
    crushes['Data'] = crushes['Data'].apply(smooth_force)
    crushes['Data'] = crushes['Data'].apply(add_pressure)
    crushes['Data'] = crushes['Data'].apply(add_stress)
    crushes['Data'] = crushes['Data'].apply(add_strain)
    crushes['Data'] = crushes['Data'].apply(add_stiffness)
    return crushes


def calculate(crushes):
    """
    Adds calculated statistics about each crush transient and returns
    Suggest running modify() first to get expected results
    """

    def to_stress(force):
        pin_area = np.pi * (PIN_DIAM / 2) ** 2
        return force / pin_area

    def to_strain(delta, length):
        delta, length = abs(delta), abs(length)
        return delta / length  # compressive positive

    for i, num in enumerate(crushes.index):
        crush = crushes.loc[num, 'Data']

        # Tissue thickness
        thickness = abs(contact_position(crush))
        crushes.loc[num, 'Thickness (mm)'] = thickness

        # Crush duration
        crushes.loc[num, 'Crush Duration (s)'] = crush_duration(crush)

        # Target duration
        delta = target_duration(crush)
        crushes.loc[num, 'Target Duration (s)'] = delta

        # Target stress
        target_stress = crush.loc[target_time(crush), 'Stress (MPa)']
        crushes.loc[num, 'Target Stress (MPa)'] = target_stress

        # Target strain
        target_strain = crush.loc[target_time(crush), 'Strain']
        crushes.loc[num, 'Target Strain'] = target_strain

        # Stiffness at contact
        # Assumed to be minimum
        stiffness = crush['Stiffness (MPa)'].min()
        crushes.loc[num, 'Contact Stiffness (MPa)'] = stiffness

        # Stiffness at target
        # Assumed to be maximum
        stiffness = crush['Stiffness (MPa)'].max()
        crushes.loc[num, 'Target Stiffness (MPa)'] = stiffness

        # Delta stress after target reached
        stress_relaxation = to_stress(target_relaxation(crush))
        crushes.loc[num, 'Relaxation Stress (MPa)'] = stress_relaxation

        # Delta strain after target reached
        holding_strain = to_strain(target_movement(crush), thickness)
        crushes.loc[num, 'Holding Strain'] = holding_strain

    return crushes


def preprocess(crushes, targets):
    """
    Adds targets for each crush available, removes non-features,
    encodes the categorical feature labels and returns X, y
    Note that crushes gets modified (side affect)
    """

    # Init new features
    crushes['Pathologist'] = np.nan
    crushes['Serosal Thickness (mm)'] = np.nan
    crushes['Post Serosal Thickness (mm)'] = np.nan
    crushes['Serosal Thickness Change (mm)'] = np.nan

    # Get list of features and targets
    excluded = ['Test ID',
                'Patient',
                'Load (g)',
                'Summary',
                'Data']
    feature_names = list(crushes.columns)
    for ex in excluded:
        if ex in feature_names:
            feature_names.remove(ex)
    target_names = ['Trauma Score', 'P Score']

    # Init new targets
    for name in target_names:
        crushes[name] = np.nan

    # Match targets to crushes, last column is targets
    for num in targets.index:

        # Identify matching crushes by four criteria
        sel = {'PROT': targets.loc[num, 'Protocol'],
               'CODE': targets.loc[num, 'Patient Code'],
               'TISS': targets.loc[num, 'Tissue'],
               'LOAD': targets.loc[num, 'Load (g)']}
        mask = crushes['Protocol'] == sel['PROT']
        mask = mask & (crushes['Patient'] == sel['CODE'])
        mask = mask & (crushes['Tissue'] == sel['TISS'])
        mask = mask & (crushes['Load (g)'] == sel['LOAD'])
        assert mask.sum() == 1, f"Matching error: {mask.sum():d} matches found"

        # Assign new feature values
        crushes.loc[mask, 'Pathologist'] = targets.loc[num, 'Pathologist']
        delta = targets.loc[num, 'Absolute Delta (um)'] / 1000
        percent_delta = targets.loc[num, 'Percent Delta'] / 100
        thickness = delta / percent_delta
        crushes.loc[mask, 'Serosal Thickness (mm)'] = thickness
        crushes.loc[mask, 'Post Serosal Thickness (mm)'] = thickness - delta
        crushes.loc[mask, 'Serosal Thickness Change (mm)'] = delta

        # Add actual targets
        for name in target_names:
            crushes.loc[mask, name] = targets.loc[num, name]

    # Make a copy of features and targets removing any without pathology rating
    valid = ~crushes['Trauma Score'].isna()
    X = crushes.loc[valid, feature_names].copy()
    y = crushes.loc[valid, target_names].copy()

    # One hot encode categorical variables
    categorical = list(X.dtypes[X.dtypes == 'object'].index)
    legend = {}
    renames = {}
    for label in categorical:
        cats = X[label].unique()
        if label == 'Patient':
            idx_cats = [np.int64(c[2:]) for c in cats]
        elif len(cats) == 1:  # constant features not useful
            legend[f"{label}[{cats[0]}]"] = '(excluded)'
            X = X.drop(label, axis=1)
            continue
        else:
            assert len(cats) == 2, "Unknown categorical feature"
            idx_cats = [False, True]

        for i, cat in enumerate(cats):
            idx = idx_cats[i]
            legend[f"{label}[{cat}]"] = str(idx)
            X.loc[X[label] == cat, label] = idx

        renames[label] = f"{label} ({cats[0]} or {cats[1]})"
    X = X.rename(renames, axis=1)

    return X, y, legend


def classify(y, drop_cols=True):
    '''
    Input the target values and change them to be boolean and more descriptive.
    '''
    y['Significant Serosal Change'] = y['P Score'] < 0.05  # 5% significance
    y.loc[y['P Score'].isna(), 'Significant Serosal Change'] = np.nan

    y['Tissue Damage'] = y['Trauma Score'] > 0
    y.loc[y['Damage Score'].isna(), 'Tissue Damage'] = np.nan

    y['Major Tissue Damage'] = y['Trauma Score'] > 1
    y.loc[y['Damage Score'].isna(), 'Major Tissue Damage'] = np.nan

    valid = ~(y['P Score'].isna() | y['Trauma Score'].isna())
    y = y.loc[valid, :]

    if drop_cols:
        y = y.drop('P Score', axis=1)
        y = y.drop('Trauma Score', axis=1)

    return y


# MAIN
if __name__ == "__main__":
    study = study_outline(PATH)
    targets = study_targets(PATH)
    crushes = study_data(study)
    crushes = split(crushes)
    crushes = modify(crushes)
    crushes = calculate(crushes)
    X, y, legend = preprocess(crushes, targets)
    y = refine(y)
