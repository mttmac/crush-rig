#!/usr/bin/env python

# Written by: Matt MacDonald & Sachin Doshi
# For CIGITI at the Hospital for Sick Children Toronto
# Tested on LCA50-025-72F actuator

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
