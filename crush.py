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
    rig.home()
    prep(rig)
    return rig


def prep(rig):
    rig.wait(for_stop=True, chain=True)
    rig.set_mode('travel', chain=True)
    rig.move_clear(start_height)


def disconnect(rig):
    rig.set_mode('travel')
    rig.move_clear(5, wait_for_stop=True)
    rig.set_mode('safe')
    rig.move_clear(0, wait_for_stop=True)
    rig.close()


def convert_force(voltage):
    # Formula used to convert sensor data voltage reading into Newtons as per
    # predetermined sensor calibration curve
    # Calibration curve Oct 24 2018 (g - voltage)
    # 0 - 298.7
    # 100 - 334.2
    # 300 - 405.9
    # 500 - 477.5
    # 800 - 585.1
    # 1000 - 656.5
    # 1200 - 728.5
    # R^2 = 1
    # y = 0.0274x - 8.175
    return round(0.0274 * int(voltage) - 8.175, 6)


def single_crush(target_force, target_action='stop', duration=10,
                 start_time=None, multi=False):
    """
    Will execute a crush until target force is met, then will either 'stop'
    or 'hold' for duration. Logs data throughout until returned to start.
    """

    # Settings
    data = []
    window = 3
    forces = deque([0], maxlen=window)
    force_res_limit = max(0.02 * target_force, 0.1)  # aim for +/-1% error
    crush_velocity = 1.0  # mm/s
    min_velocity = crush_velocity * (2 ** -3)
    pos_margin = 0.1  # mm
    # Knockdown force extrapolation by factor if stopping at target
    if target_action == 'stop':
        knockdown = 0.25
    else:
        knockdown = 1

    # rig is at start height prior to protocol
    rig.set_mode('action')
    start_pos = rig.read_position()
    if start_time is None:
        start_time = time.time()

    # Start moving
    rig.move_const_vel(toward_home=True)

    stage = 0  # 0 for approach, 1 for crush, 2 for target, 3 for release
    done = False
    while not done:
        samples = rig.read_movement_and_force()
        samples[2] = convert_force(samples[2])
        forces.append(samples[2])

        if stage == 0:
            if sum(forces) / window >= 0:  # contact
                rig.set_max_velocity(crush_velocity)
                print('Tissue contact made..')
                stage += 1

        elif stage == 1:
            delta_force = forces[-1] - forces[-2]
            # Try to predict next value if stopped now
            if forces[-1] >= target_force - (delta_force * knockdown):
                if target_action == 'stop':
                    rig.stop()
                elif target_action == 'hold':
                    rig.move_const_torque(samples[3])
                target_time = time.time()
                print('Target force achieved..')
                stage += 1

            # Slow down if force resolution becomes poor
            elif samples[1] > min_velocity and (abs(delta_force) >
                                                force_res_limit):
                rig.set_max_velocity(max(abs(samples[1]) / 2, min_velocity))

        elif stage == 2 and (time.time() - target_time) >= duration:
            rig.set_mode('action')
            rig.move_clear(start_height)
            print('Crush complete')
            stage += 1

        elif stage == 3:
            if abs(samples[0] - start_pos) < pos_margin:
                done = True

        data.append((round(time.time() - start_time, 6), *samples, stage))

    if multi:
        return data, target_time
    return data


def multi_crush(target_force, num_crushes=5, target_action='stop',
                duration=10, duty_cycle=0.5):
    """
    Will execute a number of crushes at a set duty cycle, will either 'stop'
    or 'hold' for duration once target force achieved. Logs data throughout.
    """

    pause = duration * ((1 - duty_cycle) / duty_cycle)
    data = []
    start_time = time.time()
    for i in range(num_crushes):
        cycle_start_time = time.time()
        new_data, last_target_time = single_crush(
            target_force, target_action, duration, start_time, True)
        data += new_data

        if i == num_crushes - 1:
            continue
        time_to_target = last_target_time - cycle_start_time
        time_since_target = time.time() - last_target_time
        delay_time = (duration + pause - time_to_target) - time_since_target
        time.sleep(max(delay_time, 0))

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


def get_selection(items, name='Item'):
    name = name[0].upper() + name[1:]
    for i, item in enumerate(items):
        print(f"{i} - {item}")
    while True:
        try:
            selection = int(input(': '))
            break
        except ValueError:
            print('Invalid number')
    warning = f"{name} {selection} not recognized"
    assert selection >= 0 and selection < len(items), warning
    return items[selection]


def init(rig=None, debug=False):

    # User can input existing rig connection if available
    if type(rig) is LAC1 and not debug:
        prep(rig)
    else:
        # Connect to new rig over serial otherwise
        from serial.tools import list_ports
        port_options = [option.device for option in list_ports.comports()]

        print('Input serial port number to connect:')
        port = get_selection(port_options, 'Port')
        rig = connect(port, silent=(not debug))
        return rig


def crush(rig=None):

    # Initialize rig
    rig = init(rig)

    # Get crush settings from user
    protocol_names = ('stop', 'hold', 'multi_stop', 'multi_hold',
                      'long_stop', 'no_stop')
    print('Select a protocol:')
    protocol = get_selection(protocol_names, 'Protocol')

    max_weight = 5000  # limit to 5 kg load
    target_weight = abs(float(input('Input target load in grams: ')))
    assert target_weight <= max_weight, "Load too high"
    target_force = to_force(target_weight)

    # Select file to write csv data
    print('Storing crush data in current directory')
    filename = f"{protocol}-{target_weight}g.csv"
    filepath = Path.cwd().joinpath(filename)

    # Prevent overwriting of files
    if filepath.is_file():
        path_no_suffix = str(Path.joinpath(filepath.parent,
                                           filepath.stem)) + '*'
        matching_files = glob(path_no_suffix)
        max_version = 1
        for file in matching_files:
            if file == str(filepath):
                continue
            max_version = max(max_version, int(Path(file).stem[-2:]))
        filepath = Path.cwd().joinpath(filepath.stem +
                                       f'-{(max_version + 1):02}' +
                                       filepath.suffix)

    # Wait for go ahead
    cmd = input("Press enter to run protocol or 'x' to exit: ")
    if cmd.strip().lower() == 'x':
        return

    # Execute crush protocol
    with filepath.open('w', newline='') as file:
        writer = csv.writer(file)

        if protocol == 'stop':
            data = single_crush(target_force)

        elif protocol == 'hold':
            data = single_crush(target_force, target_action='hold')

        elif protocol == 'multi_stop':
            data = multi_crush(target_force)

        elif protocol == 'multi_hold':
            data = multi_crush(target_force, target_action='hold')

        elif protocol == 'long_stop':
            data = single_crush(target_force, duration=60)

        elif protocol == 'no_stop':
            data = single_crush(target_force, duration=0.1)

        writer.writerow(('Timestamp (s)', 'Position (mm)', 'Velocity (mm/s)',
                         'Force (N)', 'Torque', 'Stage'))
        writer.writerows(data)


# TODO add GUI interface
# TODO does force extrapolation need more than one sample?
# TODO should release mirror the crush for better resolution stiffness curves?

# Main
if __name__ == "__main__":
    global start_height
    start_height = 20  # mm
    rig = init()

    import sys
    if '-c' in sys.argv[1:]:
        crush(rig)
