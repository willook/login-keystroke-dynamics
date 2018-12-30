import numpy as np
from keyboard import record

#if shift is pressed no reason
#iamgod

if __name__ == '__main__':
    names = ['./']
    times = []

    recorded = record(until='enter')
    
    t0 = recorded[0].time
  
    #except enter
    for i in range(len(recorded)-1):
        if recorded[i].event_type == 'down':
            if recorded[i].name == 'space':
                recorded[i].name = ' '
            if recorded[i].name == 'shift':
                if i != 0 and recorded[i-1].name == 'shift':
                    times = times[:-1]
                    continue
                else:
                    recorded[i].name = ''
            
            if recorded[i].name == 'backspace':
                times = times[:-1]
                names = names[:-1]
                
            if recorded[i].name == 'delete':
                times = times[:-1]
                names = names[:-1]
                
            else:
                times.append(recorded[i].time-t0)
                names.append(recorded[i].name)

    name = ''.join(names)
    print(name)
    print(times)
    print(len(times))
    
    np.save(name,times)
