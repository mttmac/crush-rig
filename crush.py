#!/usr/bin/env python

# Written by Matt MacDonald
# For CIGITI at the Hospital for Sick Children Toronto
# Tested on LCA50-025-72F actuator


import csv
import time
from pathlib import Path
from glob import glob
from math import pi
from lac1 import LAC1
from collections import deque


def connect(port, silent=True):
    rig = LAC1(port, silent=silent, reset=True)
    prep(rig)
    return rig


def prep(rig):
    rig.home()
    rig.wait(for_stop=True, chain=True)
    rig.set_mode('travel', chain=True)
    rig.move_clear(start_height)


def disconnect(rig):
    rig.set_mode('travel')
    rig.move_clear(5, wait_for_stop=True)
    rig.set_mode('safe')
    rig.move_clear(0, wait_for_stop=True)
    rig.close()


def single_crush(target_force, target_action='stop', duration=10, multi=False):
    """
    Will execute a crush until target force is met, then will either 'stop'
    or 'hold' for duration. Logs data throughout until returned to start.
    """

    # TODO do I need to adjust for the hanging force of the sensor?

    data = []
    window = 10
    forces = deque(maxlen=window)

    # rig is at start height prior to protocol
    rig.set_mode('crush')
    start_pos = rig.read_position_mm()
    rig.move_const_vel(toward_home=True)

    stage = 0  # 0 for crush, 1 for action, 2 for release
    done = False
    while not done:
        samples = rig.read_pos_and_force()
        print(stage)

        if stage == 0:
            forces.append(samples[1])
            if (sum(forces) / window) >= target_force:
                if target_action == 'stop':
                    rig.stop()
                elif target_action == 'hold':
                    rig.move_const_torque(samples[2])
                target_time = time.time()
                stage = 1

        elif stage == 1 and (time.time() - target_time) >= duration:
            rig.set_mode('crush')
            rig.move_clear(start_height)
            stage = 2

        elif stage == 2:
            if abs(samples[0] - start_pos) < pos_margin:
                done = True

        data.append((time.time(), *samples, stage))

    if multi:
        return data, target_time
    return data


def multi_crush(target_force, num_crushes=5, target_action='stop',
                duration=10, duty_cycle=0.5):
    """
    Will execute a number of crushes at a set duty cycle, will either 'stop'
    or 'hold' for duration once target force achieved. Logs data throughout.
    """

    pause = duration * ((1 - duty_cycle) / duty_cycle)  # TODO I don't think this is workign right, too short
    data = []
    for i in range(num_crushes):
        new_data, last_target_time = single_crush(target_force, target_action,
                                                  duration, multi=True)
        data += new_data

        if i == num_crushes - 1:
            continue
        time.sleep(min(duration + pause - (time.time() - last_target_time), 0))

    return data


def to_force(weight):
    """
    Converts weight in grams to force in N at standard earth gravity.
    """
    return 9.81 * weight / 1000


def to_pressure(force, diameter=5):
    """
    Converts force reading in N to pressure in kPa based on a circular
    diameter in mm. Default diameter is 5 mm.
    """
    area = pi * (diameter / 2) ** 2
    return 1000 * force / area


# Calibration curve July 18, 2017 # TODO update
# 0 - 248
# 200 - 294
# 300 - 317
# 500 - 363
# 800 - 432
# 1000 - 479
# 1200 - 525
# R^2 = 1

# TODO add GUI interface
# mac serial port: /dev/cu.usbserial-FTV98A40

global stage
debug = True  # turns silent false
start_height = 20  # mm
pos_margin = 0.1  # mm
protocol_names = ('stop', 'hold', 'multi_stop', 'long_stop')

# Connect to rig if not already connected
try:  # TODO this is not working in Anaconda, different namespace
    rig
except NameError:
    port = input('Input serial port name to connect: ').strip()
    rig = connect(port, silent=(not debug))
else:
    prep(rig)  # reset home

protocol = input('\n- '.join(['Select a protocol:', *protocol_names]) +
                 '\n: ').strip().lower()
assert protocol in protocol_names, 'Invalid protocol input'

target_weight = float(input('Input target load in grams: '))
target_force = to_force(target_weight)

# Select file to write csv data
print('Storing crush data in current directory')
filename = f"{protocol}-{target_weight}g.csv"
filepath = Path.cwd().joinpath(filename)

# Prevent overwriting of files
if filepath.is_file():
    path_no_suffix = str(Path.joinpath(filepath.parent, filepath.stem)) + '*'
    matching_files = glob(path_no_suffix)
    max_version = 1
    for file in matching_files:
        if file == str(filepath):
            continue
        max_version = max(max_version, int(Path(file).stem[-2:]))
    filepath = Path.cwd().joinpath(filepath.stem + f'-{(max_version + 1):02}' +
                                   filepath.suffix)

with filepath.open('w', newline='') as file:
    writer = csv.writer(file)

    cmd = input("Press enter to run protocol or 'x' to exit: ")
    if cmd.strip().lower() == 'x':
        import sys
        sys.exit()

    if protocol == 'stop':
        data = single_crush(target_force, target_action='stop')

    elif protocol == 'hold':
        data = single_crush(target_force, target_action='hold')

    elif protocol == 'multi_stop':
        data = multi_crush(target_force, target_action='stop')

    elif protocol == 'long_stop':
        data = single_crush(target_force, target_action='stop', duration=60)

    writer.writerow(('Timestamp (s)', 'Position (mm)', 'Force (N)',
                     'Torque', 'Stage'))
    writer.writerows(data)
