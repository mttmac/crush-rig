#!/usr/bin/env python
import serial
import time
import csv
import matplotlib.pyplot as plt
import threading
import matplotlib.animation as animation
from pathlib import Path



# It is important to make these floats to avoid integer truncation error
ENC_COUNTS_PER_MM = 1000.0  # encoder counts per mm
SERVO_LOOP_FREQ = 5000.0    # servo loop frequency

# This is specific to the stage I am using
# TODO Implement range checking for safety?
STAGE_TRAVEL_MM = 25
STAGE_TRAVEL_UM = STAGE_TRAVEL_MM * 1000
STAGE_TRAVEL_ENC = STAGE_TRAVEL_MM * ENC_COUNTS_PER_MM

# we will not allow travel beyond TRAVEL_SAFETY_FACTOR * STAGE_TRAVEL_ENC
TRAVEL_SAFETY_FACTOR = 0.9

# KV and KA defined the change in encoder per servo loop needed to achieve
# 1 mm/s velocity and 1 mm/s/s acceleration, respectively.
KV = 65536 * ENC_COUNTS_PER_MM / SERVO_LOOP_FREQ
KA = 65536 * ENC_COUNTS_PER_MM / (SERVO_LOOP_FREQ**2)

# These parameters are dependent on the stage. See SMAC Actuators Users Manual
SG = 2
SI = 10
SD = 30
IL = 5000
SE = 16383
RI = 0
FR = 1

# Time parameter to WS commands. Unit is ms
WS_PERIOD_MS = 25

# LAC-1 manual recommends a small delay of 100 ms after sending commands; this has been reduced for higher frequency required for this study
SERIAL_SEND_WAIT_SEC = 0.005

# Each line cannot exceed 127 characters as per LAC-1 manual
SERIAL_MAX_LINE_LENGTH = 127

class LAC1(object):
  """
  Class to interface with a SMAC LAC-1 module.

  SMAC serial interface accepts instructions in the format of:
    <command>[<argument>],<command>[<argument>],... <CR>
  e.g. SG1000,SD5000 <CR>

  """
  _silent = True
  _port = None
  _ESC = '\033'
  _sleepfunc = time.sleep
  _last_serial_send_time = None

  def __init__(self, port, baudRate, silent=True, reset=True, sleepfunc=None):
    """
    If silent is True, then no debugging output will be printed.

    If sleepfunc is not None, then it will be used instead of time.sleep.
    It will be passed the number of seconds to sleep for. This is provided
    for integration with single threaded GUI applications.
    """

    if sleepfunc is not None:
      self._sleepfunc = sleepfunc

    print('Connecting to LAC-1 on %s (%s)'%(port, baudRate))
    self._port = serial.Serial(
        port = port,
        baudrate = baudRate,
        bytesize = 8,
        stopbits = 1,
        parity = 'N',
        timeout = 0.1)

    self._silent = silent
    self.sendcmds('EF', wait=False)

    # setup some initial parameters
    self.sendcmds(
        'SG', SG,
        'SI', SI,
        'SD', SD,
        'IL', IL,
        'SE', SE,
        'RI', RI,
        'FR', FR)

    # these are pretty safe values
    self.set_max_velocity(1)
    self.set_max_acceleration(1)

  def _readline(self, stop_on_prompt=True):

    #print 'reading line',
    # XXX The loop below implicitly handles timeouts b/c when c == '' due to
    # timeout, line += '' is a null op, and the loops continues indefinitely
    # until exitconditions are met

    done = False
    line = str()
    allowedtimeouts = int(30/self._port.timeout)

    while not done:
      c = self._port.read().decode("utf-8")
      if c == '\n':
        continue
      elif c == '\r':
        done = True
      elif c == '':
        allowedtimeouts -= 1
        if allowedtimeouts == 0:
          raise Exception('Read Timed Out')
      else:
        line += c
        if stop_on_prompt and c == '>':
          done = True

    if len(line) and line[0] == '?':
      raise Exception('LAC-1 Error: '+line[1:])

    if not self._silent:
      print('[>]',line)
    return line

  def sendcmds(self, *args, **kwargs):
    """
    This method sends the given commands and argument to LAC-1. Commands are
    expected in the order of
      cmd arg cmd arg

    And will be sent as:
      $cmd$arg,$cmd,$arg<CR>
    """
    # XXX enforce SERIAL_SEND_WAIT_SEC
    if self._port is None:
      return

    now = time.time()
    if self._last_serial_send_time is not None:
      dt = now - self._last_serial_send_time
      timeleft = SERIAL_SEND_WAIT_SEC - dt
      if timeleft > 0:
        self._sleepfunc(timeleft)

    if len(args) == 1:
      cmds = [args[0]]
    else:
      assert(len(args)%2 == 0)

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

        cmds.append(cmd+arg)

    tosend = ','.join(cmds)
    print(tosend)

    if not self._silent:
      print('[<]',tosend)

    self._port.flushInput()
    self._port.flushOutput()

    assert len(tosend) <= SERIAL_MAX_LINE_LENGTH, 'Command exceeds allowed line length'

    data = (tosend+'\r')
    if(type(data) == type('String')):
      byte_command = bytearray(data,'ascii')

    self._port.write(byte_command)
    #modification original: self._port.write(tosend+'\r')

    wait = kwargs.get('wait', True)
    callbackfunc = kwargs.get('callback', None)

    datalines = []

    if wait:
      done = False
      while not done and self._port is not None:
        #print 'sendcmds, reading'
        line = self._readline()
        #print 'sendcmds:',line
        if line == '>':
          done = True
        elif line is not None and len(line):
          if callbackfunc is not None:
            callbackfunc(line)
          datalines.append(line)

      # If we have more than one line, then ignore the first which is repeat
      # of what we sent due to echo been on by default.
      # XXX I don't try to disable echo because I can't seem to turn it off
      # reliably.
      if len(datalines) == 1:
        return datalines
      else:
        return datalines[1:]
    else:
      # we update _last_serial_send_time only if we are not
      # waiting for a response
      self._last_serial_send_time = now
      return None

  def set_max_velocity(self, mmpersecond):
    self.sendcmds('SV', KV*mmpersecond)

  def set_max_acceleration(self, mmpersecondpersecond):
    self.sendcmds('SA',KA*mmpersecondpersecond)

  def set_max_torque(self, q=10000):
    self.sendcmds('SQ',q)

  def close(self):
    if self._port:
      self._port.write(self._ESC)
      self._port.write(self._ESC)
      # abort, motor off, echo on
      self._port.write('AB,MF,EN\r')
      self._port.close()
      self._port = None

