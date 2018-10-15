#!/usr/bin/env python

# Modified by Matt MacDonald
# For CIGITI at the Hospital for Sick Children Toronto
# Forked from https://github.com/freespace/smac-lac-1


import time
from serial import Serial
from serial.tools import list_ports


# Constants for LCA50-025-72F actuator, change for alternative actuators
ENC_COUNTS_PER_MM = 10000.0  # encoder counts per mm
SERVO_LOOP_FREQ = 5000.0  # servo loop frequency (only when SS = 2)
STAGE_TRAVEL_MM = 25
MAX_TRAVEL_MM = 20
HOME_EXTENDED = True  # true if actuator is intended to be extended at zero
STAGE_TRAVEL_ENC = min(STAGE_TRAVEL_MM, MAX_TRAVEL_MM) * ENC_COUNTS_PER_MM
# KV and KA define the rates in encoder per servo loop needed to achieve
# 1 mm/s velocity and 1 mm/s/s acceleration, respectively.
KV = 65536 * ENC_COUNTS_PER_MM / SERVO_LOOP_FREQ
KA = 65536 * ENC_COUNTS_PER_MM / (SERVO_LOOP_FREQ ** 2)
# These parameters are dependent on the stage. See SMAC Actuators Users Manual.
SS = 2
SG = 2
SI = 10
SD = 30
IL = 5000
SE = 16383
RI = 0
FR = 1
# Time to wait for actuator to stop in ms.
WS_PERIOD_MS = 25
# LAC-1 manual recommends a small delay of 100 ms after sending commands
SERIAL_SEND_WAIT_SEC = 0.1  # TODO need to be faster?
# Each line cannot exceed 127 characters as per LAC-1 manual
SERIAL_MAX_LINE_LENGTH = 127


# Class definition using above constants
class LAC1(object):
    """
    Class to interface with a SMAC LAC-1 module.

    SMAC serial interface accepts instructions in the format of:
    <command>[<argument>],<command>[<argument>],... <CR>
    e.g. SG1000,SD5000 <CR>

    Note that EF is sent as the first command to LAC-1 on initialisation, and
    EN is sent as the last command on close to simplify parsing outputs.

    The actuator is reset (RT) on startup unless explicitly prevented.
    """

    def __init__(self, port, baudRate=19200,
                 silent=True, reset=False, sleepfunc=None):
        """
        If silent is True, then no debugging output will be printed.

        If sleepfunc is not None, then it will be used instead of time.sleep.
        It will be passed the number of seconds to sleep for. This is provided
        for integration with single threaded GUI applications.
        """

        # Store commands to chain together when communication latency
        # is a concern
        self._chain_cmds = []

        # Store last known position for travel range checking
        self._current_pos_enc = None

        if sleepfunc is not None:
            self._sleepfunc = sleepfunc
        else:
            self._sleepfunc = time.sleep
        self._silent = silent

        print(f'Connecting to LAC-1 on {port} ({baudRate})')
        port_options = []
        for option in list_ports.comports():
            port_options.append(option.device)
        assert port in port_options, 'Serial port not recognized'
        self._port = Serial(
            port=port,
            baudrate=baudRate,
            bytesize=8,
            stopbits=1,
            parity='N',
            timeout=0.1)
        self._last_serial_send_time = None

        # Reset then setup initial parameters
        if reset:
            self.sendcmds('RM,RT', wait=False)
            self._sleepfunc(3)  # wait for reset
        self.sendcmds('EF', wait=False)
        self.sendcmds(
            'SS', SS,
            'SG', SG,
            'SI', SI,
            'SD', SD,
            'IL', IL,
            'SE', SE,
            'RI', RI,
            'FR', FR)
        self.set_mode(mode='safe')  # initial movement rates
        print(f'Successfully connected to LAC-1 on {port} ({baudRate})')

    # Communication methods
    def _readline(self, stop_on_prompt=True):
        '''
        Returns a line, that is reads until \r. Note that there are
        some commands that will suppress the \r, so be careful if you
        use those commands and this method.

        If stop_on_prompt is True, and it is by default, then if we will stop
        when we consume '>', returning whatever we have read so far as a line,
        including the '>'. The method self._port.readline() is not used b/c
        the escape character can not be set.

        The loop below implicitly handles timeouts b/c when c == '' due to
        timeout, line += '' is a null op, and the loop continues indefinitely
        until exit conditions are met.
        '''

        done = False
        line = str()
        allowedtimeouts = int(30 / self._port.timeout)

        while not done:
            c = self._port.read().decode('utf-8')
            if c == '\n':
                continue
            elif c == '\r':
                done = True
            elif c == '':
                allowedtimeouts -= 1
                if allowedtimeouts == 0:
                    raise Exception('Serial read timed out')
            else:
                line += c
                if stop_on_prompt and c == '>':
                    done = True

        if len(line) and line[0] == '?':
            raise Exception(f'LAC-1 Error: {line[1:]}')

        if not self._silent:
            print('[>]', line)

        return line

    def sendcmds(self, *args, chain=False, wait=True, callback=None):
        """
        This method sends the given commands and arguments to the LAC-1.

        Commands are expected in the order of cmd, arg, cmd, arg, etc.
        And will be sent as:
            $cmd$arg,$cmd$arg<CR>
        If a command takes no argument, then put None or ''.

        If communication latency is a concern commands can be chained together
        by setting the keyword chain to True. This will delay sending the
        commands until this method is run with chain False.
        Running this method with no arguments will execute the stored
        chain commands and send them all at once.

        If wait is True, then after sending each command, the serial stream
        is consumed until '>' is encountered. This is because SMAC emits
        '>' when it is ready for another command.
        wait is True by default

        If callback is not None, and wait is True, then after reading
        each line from the LAC-1, the callback will be invoked with the
        contents of the line.
        """

        assert self._port is not None, 'Serial communication disconnected'

        if len(args) == 1:
            cmds = [args[0]]
        else:
            assert(len(args) % 2 == 0)

            args = list(args)
            cmds = []
            while len(args):
                cmd = args.pop(0)
                arg = args.pop(0)

                if arg is not None:
                    if type(arg) is float:
                        arg = int(arg)
                    arg = str(arg)
                else:
                    arg = ''
                cmds.append(cmd + arg)

        if self._chain_cmds:
            # Add the stored chain commands to send
            cmds = self._chain_cmds + cmds

        if chain:
            self._chain_cmds = cmds
            return

        # Send commands over serial connection
        now = time.time()
        if self._last_serial_send_time is not None:
            dt = now - self._last_serial_send_time
            timeleft = SERIAL_SEND_WAIT_SEC - dt
            if timeleft > 0:
                self._sleepfunc(timeleft)

        tosend = ','.join(cmds)
        if not self._silent:
            print('[<]', tosend)

        self._port.flushInput()
        self._port.flushOutput()

        assert len(tosend) <= SERIAL_MAX_LINE_LENGTH, (
            'Command exceeds allowed line length')

        self._port.write(bytearray(tosend + '\r', 'utf-8'))

        # Reset chain cmds
        self._chain_cmds = []

        datalines = []
        if wait:
            done = False
            while not done and self._port is not None:
                line = self._readline()
                if line == '>':
                    done = True
                elif line is not None and len(line):
                    if callback is not None:
                        callback(line)
                    datalines.append(line)
            return datalines
        else:
            # Enforce delay only if we didn't wait for a response
            self._last_serial_send_time = now
            return None

    # Low level motion methods
    def position_mode(self, **kwargs):
        self.sendcmds('PM', **kwargs)

    def velocity_mode(self, **kwargs):
        self.sendcmds('VM', **kwargs)

    def torque_mode(self, **kwargs):
        self.sendcmds('QM0', **kwargs)  # only voltage mode implemented

    def go(self, **kwargs):
        self.sendcmds('GO', **kwargs)

    def stop(self, **kwargs):
        self.sendcmds('ST', **kwargs)

    def wait(self, time_in_ms=WS_PERIOD_MS, for_stop=False, **kwargs):
        if not for_stop:
            self.sendcmds('WA', time_in_ms, **kwargs)
        else:
            self.sendcmds('WS', time_in_ms, **kwargs)

    def abort(self, **kwargs):
        self.sendcmds('AB', **kwargs)

    def motor_on(self, **kwargs):
        self.sendcmds('MN', **kwargs)

    def motor_off(self, **kwargs):
        self.sendcmds('MF', **kwargs)

    def set_max_velocity(self, mmpersecond, **kwargs):
        self.sendcmds('SV', KV * abs(mmpersecond), **kwargs)

    def set_max_acceleration(self, mmpersecond2, **kwargs):
        self.sendcmds('SA', KA * abs(mmpersecond2), **kwargs)

    def set_max_torque(self, torque=10000, **kwargs):
        self.sendcmds('SQ', torque, **kwargs)

    def set_direction(self, extend=True, **kwargs):
        if extend:
            direction = 0  # extend
        else:
            direction = 1  # retract
        self.sendcmds('DI' + str(direction), **kwargs)

    # General motion methods
    def set_home(self):
        '''
        Process to set home. Simply lets encoder fall under gravity and
        stores the location. More complex soft touch methods could be used
        if required.
        '''
        self.motor_off(chain=True)
        self.wait(1000, for_stop=True, chain=True)
        self.sendcmds('DH0')  # store home as 0

        self._current_pos_enc = self.read_position()
        self.motor_on()

    def home(self):
        if self._current_pos_enc is None:
            self.set_home()
        self.sendcmds('GH')

    def move_enc(self, pos_enc, relative=False, wait_for_stop=False,
                 return_pos=False):
        """
        Move to a position specified in encoder counts.
        Alternatively move in relative encoder counts.
        Can optionally cease commands until actuator stops.
        Can optionally return the position after executing the move.
        """

        # Actuator must be homed before any motion can occur
        assert self._current_pos_enc is not None, (
            "Home the actuator before attempting to move")

        # Check for travel range limits
        if relative:
            dist_enc = pos_enc
            pos_enc += self._current_pos_enc
        assert abs(pos_enc) <= STAGE_TRAVEL_ENC, (
            'Commanded to move beyond travel limits')
        if HOME_EXTENDED:
            assert pos_enc <= 0, (
                'Commanded to move beyond home limit')
        else:
            assert pos_enc >= 0, (
                'Commanded to move beyond home limit')

        self.position_mode(chain=True)
        self.motor_on(chain=True)
        if relative:
            self.sendcmds('MR', int(dist_enc), chain=True)
        else:
            self.sendcmds('MA', int(pos_enc), chain=True)
        self.go(chain=True)

        if wait_for_stop:
            self.wait(for_stop=True, chain=True)

        if return_pos:
            return self.read_position()
        else:
            self.sendcmds(wait=False)

    def move_mm(self, pos_mm, **kwargs):
        pos_enc = self.move_enc(pos_mm * ENC_COUNTS_PER_MM, **kwargs)
        if pos_enc is not None:
            return pos_enc / ENC_COUNTS_PER_MM

    def move_clear(self, dist_mm, **kwargs):
        """
        Provide clearance from home by moving a distance away from it in mm.
        """
        if HOME_EXTENDED:
            dist_mm = -abs(dist_mm)
        else:
            dist_mm = abs(dist_mm)
        return self.move_mm(dist_mm, **kwargs)

    def move_const_vel(self, mmpersecond=None, toward_home=True):
        """
        Start the actuator moving in a direction at a velocity in mm/s.
        Moves toward home at max velocity by default.
        """
        assert self._current_pos_enc is not None, (
            "Home the actuator before attempting to move")

        self.velocity_mode(chain=True)
        if toward_home:
            extend = HOME_EXTENDED
        else:
            extend = not HOME_EXTENDED
        self.set_direction(extend=extend, chain=True)
        if mmpersecond is not None:
            self.set_max_velocity(mmpersecond, chain=True)
        self.go(wait=False)

    def move_const_torque(self, torque, **kwargs):
        """
        Start the actuator moving in a direction at a velocity in mm/s.
        Moves toward home by default.
        """
        assert self._current_pos_enc is not None, (
            "Home the actuator before attempting to move")

        self.set_max_torque(torque, chain=True)
        self.torque_mode(chain=True)
        self.go(wait=False)

    def set_mode(self, mode='safe'):
        """
        Set the speed, acceleration and torque limits for the mode needed.
        Modes:
        - safe
        - travel
        - crush
        """
        if mode == 'safe':
            mmpersecond = 1
            mmpersecond2 = 100
            q = 10000
        elif mode == 'travel':
            mmpersecond = 75
            mmpersecond2 = 3800
            q = 32000
        elif mode == 'crush':
            mmpersecond = 6
            mmpersecond2 = 3000
            q = 30000
        else:
            return
        self.position_mode(chain=True)
        self.set_max_velocity(mmpersecond, chain=True)
        self.set_max_acceleration(mmpersecond2, chain=True)
        self.set_max_torque(q)

    # Read methods
    # Warning: read methods can't be chained and will end current chains
    def read_error(self):
        """
        Returns the last error from LAC-1 if there is one
        """
        error = self.sendcmds('TE')
        if error:
            return error[-1]  # assume last entry in case chained

    def read_pos_error(self):
        """
        Returns the current position offset from target in encoder counts
        """
        raw_output = self.sendcmds('TT,TP')
        self._current_pos_enc = int(raw_output[-1])
        target = int(raw_output[-2])
        return self._current_pos_enc - target

    def read_position(self):
        """
        Returns the current position in encoder counts
        """
        self._current_pos_enc = int(self.sendcmds('TP')[-1])
        return self._current_pos_enc

    def read_position_mm(self):
        """
        Returns the current position in mm
        """
        return self.read_position() / ENC_COUNTS_PER_MM

    def read_parameters(self, pset=0):
        """
        Return parameter set (0 or 1)
        """
        return self.sendcmds('TK', pset)

    def read_analog_input(self, channel=0):
        """
        Return an analog input reading from ADC (0...n)
        10 bit resolution (0 to 1024)
        """
        return int(self.sendcmds('TA', channel)[-1])

    def read_torque(self, channel=0):
        """
        Return the current torque of the actuator motor.
        """
        return int(self.sendcmds('TQ')[-1])

    def read_force(self):
        """
        Return analog input reading from ADC channel 8 with attached load cell.
        Converts reading to force based on calibration table.
        """
        return self.convert_force(self.read_analog_input(8))

    def convert_force(self, voltage):
        # Formula used to convert sensor data into Newtons as per predetermined
        # sensor calibration curve
        return (9.81 * ((4.3308 * int(voltage)) - 1073.1) / 1000)

    def read_weight(self):
        """
        Converts force reading to grams at standard earth gravity.
        """
        return 1000 * self.read_force() / 9.81

    def read_pos_and_force(self):
        """
        Combines two simultaneous reads: position and force, to allow chaining.
        Torque is also read as an indirect metric of force (units arbitrary).
        Return units are position in mm and force in N.
        """
        raw_output = self.sendcmds('TP,TA8,TQ')
        self._current_pos_enc = int(raw_output[-3])
        return (self._current_pos_enc / ENC_COUNTS_PER_MM,
                self.convert_force(raw_output[-2]), int(raw_output[-1]))

    # Shutdown methods
    def close(self):
        if self._port is not None:
            # Send escape char twice
            for _ in range(2):
                self._port.write(bytearray('\033', 'utf-8'))

            # Abort, motor off, echo on
            self.sendcmds('AB,MF,EN')

            self._port.close()
            self._port = None
