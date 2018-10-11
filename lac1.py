#!/usr/bin/env python
import serial
import time
import csv
import matplotlib.pyplot as plt
import threading
import matplotlib.animation as animation
import threading
from pathlib import Path

# Modified for LCA50-025-72F actuator model
# for CIGITI at the Hospital for Sick Children Toronto
# Sachin Doshi & Matt MacDonald

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

# Tests #####################################################################
def test_set_home_macro():
  #lac1 = LAC1('/dev/ttyS0', 19200, silent=False)
  lac1 = LAC1('COM3', 19200, silent=False)
  lac1.set_home_macro(force=True)
  lac1.home()
  p = lac1.get_position_enc()
  print(p)
  #assert abs(p) <= 10, p

def test_home():
  lac1 = LAC1('COM3', 19200, silent=False)
  lac1.home()
  p = lac1.get_position_enc()
  print(p)
  #assert abs(p) <= 10, p
    

def constant_force(user_input_target_grams):
  
  #Adjustable variables for this load mode
  force_hold_time = 20       #How long to hold to keep applying pressure after the target pressure is achieved for the first time (as identified by moving average)
  enter_target_pressure_grams = user_input_target_grams   #Target pressure for load mode; identified by moving average
  
  
  
  
  target_pressure_grams = ((enter_target_pressure_grams-40)) #the -40 is to account for the weight of the load cell itself, which is not registered in these readings.
  
  force_error_percent = 0.03   #This is the % difference from target force that will yeild no movement in the actuator (i.e. will be ignored) when the moving average is calculated
  error_percent_for_adjust_motor = 0.80   #Once the force is within this percentage of the target force, the actuator movements will be decreased to 'adjust motor step' (see below)
  
  #Variables for moving average
  loop_number = 0   #Used to run code & write to the moving average; *Do not adjust*
  size_of_moving_average = 2   #How many data points in the moving average *Adjust this value* Note that for constant force, this value should be relatively low (when compared to the target threshold and hold position) since the movement of the actuator will be based on this and should move dynamically with changes in force 
  running_total_force = [0]*size_of_moving_average  #This is to create a list with # of elements = moving average
  
  #Variables for running this load mode *Do not adjust*
  time_under = True   #While this is true, the actuator will be holding pressure, reading, and writing data; this becomes false when the test time reaches its limit (this is in the code)
  force_converted = 0   #Defining a variable outside of the load mode loop, set to 0 by default
  load_mode_start_time = 9999999999999999999  #This is set to 0 to trigger an if loop later in the code  
  
  
  while (time_under):

    motor_step = 2000  #The distance in um that the actuator moves with each command sent (coordinating with the sleep time can be used to calibrate velocity) 
    #motor step of 2000 =~ 4mm/s speed
    
    #motor_step = int(400*(1-(0.667*(force_converted/target_pressure_grams))))    #This equation adjusts actuator movements decreasingly as it approaches the target pressure (note that this also decreases velocity)
    adjust_motor_step = 10    #Once the target force is achieved, this is the distance in um that the actuator moves each time it detects that the force on the load cell is outside of the error threshold
    
    if (force_converted > (target_pressure_grams*error_percent_for_adjust_motor)):    #If the force is within a certain percent of the target, the actuator movements will become more fine
      motor_step = adjust_motor_step
    
    moving_average_force = (sum(running_total_force)/len(running_total_force))        #Calculates the moving average that will be updated via the loop below
    
    if (moving_average_force < (target_pressure_grams*(1-force_error_percent))):      #If the moving average of force is below the target threshold then move the actuator forwards. Report force & position data.
      #position_and_force = lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR%s,GO,WS,TP,TP,TA8,WA1' %(motor_step))
      position_and_force = lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR%s,GO,TP,TP,TA8,WA1' %(motor_step))
    elif (moving_average_force > (target_pressure_grams*(1+force_error_percent))):    #If the moving average of force is below the target threshold then move the actuator backwards. Report force & position data.
      #position_and_force = lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR-%s,GO,WS,TP,TP,TA8,WA1' %(motor_step))    
      position_and_force = lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR-%s,GO,TP,TP,TA8,WA1' %(motor_step))    
      
    else: #This means that we are within the force target window                      #If the moving average of force is within the error range, then hold position. Report force & position data.
      position_and_force = lac1.sendcmds('TP,TP,TA8,WA1')  
      if (load_mode_start_time == 9999999999999999999):
        load_mode_start_time = time.clock() #This is when the target pressure is reached and so a timer is started; after it is set, it is no longer equal to 0 and so this loop is avoided / the clock is not reset
      
    relative_time = (time.clock() - ref_time)
    constant_force_time = (time.clock() - load_mode_start_time)    #This is will trigger a cut-off after X seconds of constant force
    
    force_converted = ((4.3308*int(position_and_force[1]))-1073.1) #Formula used to convert sensor data into grams/Newtons/kPa (predetermined and calibrated using the sensor), currently in GRAMS *Do not adjust this formula*
    data = [relative_time,position_and_force[0],force_converted]
    writer.writerow(data)
    
    #Recording data to the moving average and moving to the next index for the next data point
    running_total_force[(loop_number%size_of_moving_average)] = force_converted
    loop_number += 1        
    
    if (constant_force_time>force_hold_time):
      data =["Constant force achieved at ",str(relative_time - force_hold_time),("time %s s, grams %s g, force error percent: %s, mvg_avg %s, motor step %s, adjust motor step %s, cutoff for adjust motor step rate at %s percent." %(force_hold_time,target_pressure_grams, force_error_percent, size_of_moving_average,motor_step,adjust_motor_step,error_percent_for_adjust_motor))]
      writer.writerow(data)      
      time_under = False 
      lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR-270000,GO')           
      
  return 0
        
        
        
def constant_position(user_input_target_grams):
  
  #Adjustable variables for this load mode
  position_hold_time = 20       #How long to hold a position after the desired force has been met
  enter_target_pressure_grams = user_input_target_grams   #Target pressure for load mode; identified by moving average
 
 
 
    
  target_pressure_grams = ((enter_target_pressure_grams-40)-(enter_target_pressure_grams*.05)) #the -40 is to account for the weight of the load cell itself, which is not registered in these readings. The subtraction of 5% of the end target pressure is in place to reduce when movement is cut off, which will reduce the pressure overshoot
  
  #Variables for moving average
  loop_number = 0   #Used to run code & write to the moving average; *Do not adjust*
  size_of_moving_average = 1   #How many data points in the moving average *Adjust this value*
  running_total_force = [0]*size_of_moving_average  #This is to create a list with # of elements = moving average
  
  #Variables for running this load mode *Do not adjust*
  time_under = True   #While this is true, the actuator will be holding pressure, reading, and writing data
  force_converted = 0   #Defining a variable outside of the load mode loop, set to 0 by default
  push_forward = True #While this is true, the actuator has not reached the target pressure and will continue to move down    
  load_mode_start_time = 9999999999999999999  #This is set to 0 to trigger an if loop later in the code  
  
  while (time_under):
    
    motor_step = 2500  #3000 =~ 4.5mm/s; 2500 =~4.00mm/s
    #motor_step = int(4000*(1-(0.667*(force_converted/target_pressure_grams))))
        
    if (force_converted > (target_pressure_grams*0.90)):
      motor_step = 100    
    
    moving_average_force = (sum(running_total_force)/len(running_total_force))
    if (moving_average_force > target_pressure_grams):
      push_forward = False  #The target pressure (via moving average) has been reached so the actuator will now be locked in place for the duration of the test
      
      if(load_mode_start_time == 9999999999999999999):
        lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR50,GO')
        load_mode_start_time = time.clock() #This is when the target pressure is reached and so a timer is started

    #Once the desired pressure is achieved, as determined by the moving average, actuator motion stops and only position/force data are measured
    if push_forward:
      #position_and_force = lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR%s,GO,WS,TP,TP,TA8,WA5' %(motor_step))
      position_and_force = lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR%s,GO,TP,TP,TA8,WA5' %(motor_step))
    else:
      position_and_force = lac1.sendcmds('MN,TP,TP,TA8,WA5')    

    constant_position_time = (time.clock() - load_mode_start_time)    #This is for cut-off after X seconds of constant position
    relative_time = (time.clock() - ref_time)               #This is for time logging purposes   
    
    force_converted = ((4.3308*int(position_and_force[1]))-1073.1) #Formula used to convert sensor data into grams/Newtons/kPa (predetermined and calibrated using the sensor), currently in GRAMS *Do not adjust this formula*
    data = [relative_time,position_and_force[0],force_converted]
    writer.writerow(data)   #Recording data
    
    #Recording data to the moving average and moving to the next index for the next data point
    running_total_force[(loop_number%size_of_moving_average)] = force_converted
    loop_number += 1    

    
    if (constant_position_time>position_hold_time):
      data =["Constant force achieved at ",str(relative_time - position_hold_time),("time %s s, grams %s g, mvg_avg %s, motor step %s. USED FLAT HEAD, not jaw geometry" %(position_hold_time,target_pressure_grams,size_of_moving_average, motor_step))]
      writer.writerow(data)            
      time_under = False 
      lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR-270000,GO')
      
  return 0



def multiple_crush(user_input_target_grams, user_input_number_of_crushes, user_input_time_per_crush,user_input_time_between_crushes):
  
  #Adjustable variables for this load mode
  enter_target_pressure_grams = user_input_target_grams   #Target pressure for load mode; identified by moving average  
  number_of_crushes = user_input_number_of_crushes     #Number of repeat crushes 
  time_per_crush = user_input_time_per_crush       #How long to hold a position after the desired force has been met
  time_between_crushes = user_input_time_between_crushes #Number of seconds released before re-crushing the tissue
  
  
  
  
  
  target_pressure_grams = ((enter_target_pressure_grams-40)-(enter_target_pressure_grams*.05)) #the -40 is to account for the weight of the load cell itself, which is not registered in these readings. The subtraction of 5% of the end target pressure is in place to reduce when movement is cut off, which will reduce the pressure overshoot
  
  #Variables for moving average
  loop_number = 0   #Used to run code & write to the moving average; *Do not adjust*
  size_of_moving_average = 1   #How many data points in the moving average *Adjust this value*
  running_total_force = [0]*size_of_moving_average  #This is to create a list with # of elements = moving average
  
  #Variables for running this load mode *Do not adjust*
  time_under = True   #While this is true, the actuator will be holding pressure, reading, and writing data
  force_converted = 0   #Defining a variable outside of the load mode loop, set to 0 by default
  push_forward = True #While this is true, the actuator has not reached the target pressure and will continue to move down    
  load_mode_start_time = 9999999999999999999  #This is set to n to trigger an if loop later in the code  
  number_of_crushes_remaining = number_of_crushes #Used to interate through the number of crushes
  data_bank = []   #Used to store then print 'constant position acheived at'
  
  while (time_under):
    
    motor_step = 3000
    #motor_step = int(400*(1-(0.667*(force_converted/target_pressure_grams))))
        
    if (force_converted > (target_pressure_grams*0.80)):
      motor_step = 25   
    
    moving_average_force = (sum(running_total_force)/len(running_total_force))
    if (moving_average_force > target_pressure_grams):
      push_forward = False  #The target pressure (via moving average) has been reached so the actuator will now be locked in place for the duration of the test
      
      if(load_mode_start_time == 9999999999999999999):
        load_mode_start_time = time.clock() #This is when the target pressure is reached and so a timer is started

    #Once the desired pressure is achieved, as determined by the moving average, actuator motion stops and only position/force data are measured
    if push_forward:
      #position_and_force = lac1.sendcmds('PM,SA20000,SV100000000,SQ15000,MR%s,GO,WS,TP,TP,TA8,WA5' %(motor_step))
      position_and_force = lac1.sendcmds('PM,SA20000,SV100000000,SQ15000,MR%s,GO,TP,TP,TA8,WA5' %(motor_step))
      
    else:
      position_and_force = lac1.sendcmds('MN,TP,TP,TA8,WA1')    

    constant_position_time = (time.clock() - load_mode_start_time)    #This is for cut-off after X seconds of constant position
    relative_time = (time.clock() - ref_time)               #This is for time logging purposes   
    
    force_converted = ((4.3308*int(position_and_force[1]))-1073.1) #Formula used to convert sensor data into grams/Newtons/kPa (predetermined and calibrated using the sensor), currently in GRAMS *Do not adjust this formula*
    data = [relative_time,position_and_force[0],force_converted]
    writer.writerow(data)   #Recording data
    
    #Recording data to the moving average and moving to the next index for the next data point
    running_total_force[(loop_number%size_of_moving_average)] = force_converted
    loop_number += 1    

    
    if (constant_position_time>time_per_crush):
      number_of_crushes_remaining += (-1)
      
      if(number_of_crushes_remaining > 0):
        load_mode_start_time = 9999999999999999999
        data_bank.append(str(relative_time - time_per_crush))
        lac1.sendcmds('PM,SA25000,SV50000000,SQ10000,MR-35000,GO')
        
        time_off_sample = time.clock()
        time_off_sample_under = True
        
        while(time_off_sample_under):
          if((time.clock() - time_off_sample) > (time_between_crushes-1)):
            time_off_sample_under = False
          position_and_force = lac1.sendcmds('MN,TP,TP,TA8,WA1')
          relative_time = (time.clock() - ref_time)
          force_converted = ((4.3308*int(position_and_force[1]))-1073.1)
          running_total_force[(loop_number%size_of_moving_average)] = force_converted
          loop_number += 1
          data = [relative_time,position_and_force[0],force_converted]
          writer.writerow(data)                   
            
        time_off_sample = 9999999999999999999
        push_forward = True
        
      else:      
        lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR-15000,GO')
        time_off_sample = time.clock()
        time_off_sample_under = True
        
        while(time_off_sample_under):
          if((time.clock() - time_off_sample) > (time_between_crushes-1)):
            time_off_sample_under = False
          position_and_force = lac1.sendcmds('MN,TP,TP,TA8,WA1')
          relative_time = (time.clock() - ref_time)
          force_converted = ((4.3308*int(position_and_force[1]))-1073.1)
          running_total_force[(loop_number%size_of_moving_average)] = force_converted
          loop_number += 1     
          data = [relative_time,position_and_force[0],force_converted]
          writer.writerow(data)                     
        
        time_string = "Constant positions achieved at: "
        
        for desired_time in data_bank:
          time_string = (time_string + desired_time + ", ")
        
        time_string = (time_string + "and " + str(relative_time - time_per_crush))
        data =[time_string,"Crush settings info:",("time %s s, grams %s g, mvg_avg %s, motor step %s, time of crush %s s, time between crushes %s s, USED FLAT HEAD not jaw geometry." %(time_per_crush,target_pressure_grams,size_of_moving_average, motor_step,time_per_crush,time_between_crushes))]
        writer.writerow(data)
        time_under = False 
        lac1.sendcmds('PM,SA25000,SV1000000000,SQ10000,MR-270000,GO')
      
  return 0

# Calibration curve July 18, 2017
# 0 - 248
# 200 - 294
# 300 - 317
# 500 - 363
# 800 - 432
# 1000 - 479
# 1200 - 525
# R^2 = 1

lac1 = LAC1('COM3', 19200, silent=False)

def MODIFY_FILE_NAME_AND_PATH_HERE():
  a=1
  return 0

file_name = "Patient1/constant_force300g.csv"

my_file = Path(file_name)
if my_file.is_file():
  input("File exists, are you sure you want to continue?")
  input("Are you CERTAIN you want to OVERWRITE this file?")
else:
  print("File does not exist, creating file.")   

write_file = open(file_name,'w',newline='')
writer = csv.writer(write_file)

lac1.sendcmds('RM')     #clears all macros from the controller
lac1.sendcmds('DH0')    #Temporarily defines positional home (setting it to 0)
#lac1.sendcmds('PM,MN,SA1000,SV100000000,SQ10000,MR-270000,GO,WS,WA500,DH0,TP')

lac1.sendcmds('PM,MN,SA25000,SV1000000000,SQ10000,MR-250000,GO,WS,WA1000')    #Actuator is retracted to allow for specimen to be placed on the rig
lac1.sendcmds('MN,WA1000,DH0')     
input("Press enter to start routine")
lac1.sendcmds('DH0')     
lac1.sendcmds('WA2000')

#lac1.sendcmds('CN1')


ref_time = time.clock()   #This is the time at which the load case begins

def MODIFY_LOAD_MODE_BELOW():
  a = 1
  #options: 
  #constant_force(target grams) ;; 
  #constant_position(target grams) ;; 
  #multiple_crush(target grams, number of crushes, time crushing (in sec), time between crushes (in sec))
  return 0

#AMY MODIFY THIS BELOW:
constant_force(300)
#constant_position(400)
#multiple_crush(500,5,10,10,10)

write_file.close()
