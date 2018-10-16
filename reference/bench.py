#!/usr/bin/env python

import sys
import time
from lac1 import LAC1


def getport():
    import os
    serialport = os.environ.get('LAC1_PORT', '/dev/ttyS0')
    return serialport


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(sys.argv[0], '<abs position in mm>')
    else:
        lac1 = LAC1(getport(), baudRate=19200)
        lac1.set_max_velocity(50)
        lac1.set_max_acceleration(100)
        p = float(sys.argv[1])
        print('Moving to', p, 'mm')
        lac1.move_absolute_mm(p)
        print('Done')
        sys.exit(1)


lac1 = LAC1(getport(), baudRate=19200)
lac1.set_home_macro(force=True)
lac1.home()

lac1.set_max_velocity(1000)
lac1.set_max_torque(10000)
lac1.set_max_acceleration(30000)
# lac1.set_max_acceleration(5000)

nloops = 1000
dist = 2

lac1.move_absolute_mm(0)
starttime = time.time()
for cnt in xrange(nloops):
    try:
        lac1.move_absolute_mm(dist, wait=False)
        p = lac1.get_position_mm()
        while p < dist:
            p = lac1.get_position_mm()

        lac1.move_absolute_mm(0, wait=False)
        p = lac1.get_position_mm()
        while p > 0:
            p = lac1.get_position_mm()

        sys.stdout.write('.')
        sys.stdout.flush()
        if cnt % 100 == 0:
            print cnt
    except Exception, ex:
        print('Exception occurred on loop %d' % (cnt + 1), ex)
        break

dt = time.time() - starttime

# we cover 2*dist per loop
disttravelled = nloops * dist * 2

print('Travelled ', disttravelled, 'mm')
print('Loops:', nloops, 'Loop distance:', dist * 2)
print('total time: %.2f\tavg speed: %.2f mm/s'
      % (dt, disttravelled / dt))
