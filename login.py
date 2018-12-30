import numpy as np
from rss import rss
from glob import glob
from keyboard import record as recording
from time import time
from os import remove

class KeyStroke():
    file_path = None
    tag_name = 'tag.txt'
    n_patterns = 10
    n_pattern = 0
    n_current = 0
    
    def __init__(self,file_path, tag_name = 'tag.txt', n_patterns = 10, n_pattern = 0, n_current = 0):
        self.file_path = file_path
        self.n_patterns = n_patterns
        self.n_current = n_current
        self.n_pattern = n_pattern    
        self.tag_name = tag_name
        
    def _record_to_time(self, records):
        names = []
        times = []
        
        t0 = records[0].time

        #마지막 인자는 엔터이므로 체크하지 않음
        for i in range(len(records)-1):
            if records[i].event_type == 'down':
                if records[i].name == 'space':
                    records[i].name = ' '
                if records[i].name == 'shift':
                    if i != 0 and records[i-1].name == 'shift':
                        times = times[:-1]
                        continue
                    else:
                        records[i].name = ''
                
                if records[i].name == 'backspace':
                    times = times[:-1]
                    names = names[:-1]
                    
                if records[i].name == 'delete':
                    times = times[:-1]
                    names = names[:-1]
                    
                else:
                    times.append(records[i].time-t0)
                    names.append(records[i].name)
        name = ''.join(names)
        return times, name
    
    def _typing_check(self, name):
        f = open(self.file_path+self.tag_name)
        line = f.readline()
        return line == name

    def _load_data(self):
        patterns = glob(self.file_path+'*.npy')
        
        self.n_current = len(patterns)
        x_times = np.zeros((self.n_patterns, self.n_pattern))
        
        for i in range(len(patterns)):
            x_time = np.load(self.file_path + patterns[i])
            x_times[i] = x_time
        return x_times

    def _recognition(self, X, y):


        pass

    

    def _update(self, time):
        
        pass
    
    def login(self):

        while True:
            records = recording(until='enter')
            y_time, y_name = _record_to_time(records)

            #입력이 태그와 같은지 확인
            if self._typing_check(y_name):
                break
            else:
                print("[!] 입력이 일치하지 않습니다")
        
        #참조 패턴을 불러옴
        self.n_pattern = len(y_time)
        x_times = _load_data()
        '''
        det = _recognition(x_times, y_time)
        if det:
            print("login 성공")
            self._update(y_time)

        else det:
            print("login 실패")
        '''

if __name__ == '__main__':
    
    k1 = KeyStroke('./patterns/')
    k1.login()
