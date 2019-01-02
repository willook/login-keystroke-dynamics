import numpy as np
from rss import rss
from glob import glob
from keyboard import record as recording
from time import time
from os import remove
from os import system
import sys
import csv

class KeyStroke():
    #file_path = None
    #tag_name = 'tag.txt'
    #n_patterns = 10
    #n_pattern = None
    #n_current = None
    
    def __init__(self,file_path, tag_name = 'tag.txt', n_patterns = 20, n_pattern = None, n_current = None, debug = False):
        self.file_path = file_path
        self.n_patterns = n_patterns
        self.n_current = n_current
        self.n_pattern = n_pattern
        self.n_valid = int(n_patterns * 0.3)
        self.tag_name = tag_name
        self.threshold = 1.4
        
        self.debug = debug

        
    def _record_to_time(self, records):
        names = []
        times = []

        if records[0].name == 'enter' and records[0].event_type == 'up':
            records = records[1:]
        
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
    
    def _record_to_time_another(self, records):
        times_2 = []

        for i in range(len(records) - 1):
            for j in range(i + 1, len(records) - 1):

                if records[i].event_type == 'down':
                    if records[i].name == '':
                        if records[j].event_type == 'up':
                            if records[j].name == '':
                                times_2.append(records[j].time - records[i].time)
                                break

                    else:
                        if records[j].event_type == 'up':
                            times_2.append(records[j].time - records[i].time)
                            break

        return times_2
    
    def _typing_check(self, name):
        f = open(self.file_path+self.tag_name)
        line = f.readline()
        return line == name

    def _load_data(self):
        
        patterns = glob(self.file_path+'*.csv')
        self.n_current = len(patterns)
        x_times1 = np.zeros((self.n_current, self.n_pattern))
        x_times2 = np.zeros((self.n_current, self.n_pattern))
        
        for i in range(len(patterns)):
            x_time1 = []
            x_time2 = []
            
            with open(patterns[i], newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    x_time1.append(row['time1'])
                    x_time2.append(row['time2'])
            x_times1[i] = x_time1
            x_times2[i] = x_time2
            
        
        return x_times1,x_times2

    def _preprocess(self,x):
        if len(x.shape) == 1:
            x_copy = np.zeros(len(x)-1)
            for i in range(len(x)-1,0,-1):
                x_copy[i-1] = x[i] - x[i-1]
                x_copy[i-1] = np.log(x_copy[i-1])
            
        if len(x.shape) == 2:
            x_copy = np.zeros((x.shape[0],x.shape[1]-1))
            for i in range(len(x)):
                for j in range(len(x[i])-1,0,-1):
                    x_copy[i][j-1] = x[i][j] - x[i][j-1]
                    x_copy[i][j-1] = np.log(x_copy[i][j-1])
                
        return x_copy

    def _weights(self,X):
        X_T = np.transpose(X)
        weights = np.ones(len(X_T))
        for i in range(len(X_T)):
            weights[i] = 1/(np.var(X_T[i][:-self.n_valid])**0.5)
        w_sum = np.sum(weights)
        for i in range(len(X_T)):
            weights[i] = weights[i]/w_sum
        return weights
        
    def _recognition(self, X, y):
        #print(X*10)
        #print(y*10)
        #print()
        
        Xp = self._preprocess(X)
        yp = self._preprocess(y)
        weights = self._weights(Xp)

        if self.debug:
            print('[weight]')
            print((weights*100).astype(np.int32))
            #print('[X times processed]')
            #print((Xp*1000).astype(np.int32))
            #print('[y time processed]')
            #print((yp*1000).astype(np.int32))

        score = 0

        if self.n_current < self.n_patterns * 0.5:
            print("data가 충분치 않습니다")
            return True

        for i in range(self.n_current - self.n_valid, self.n_current):

            min_score = self._get_diff(Xp[:-self.n_valid],Xp[i],weights)
            
            score += min_score / self.n_valid
            
        input_score = self._get_diff(Xp[:-self.n_valid],yp,weights)
        if self.debug:
            print("[input_score] :",input_score)
            print("[threshold] :",score * self.threshold)
        if input_score <= score * self.threshold:
            return True
        
        return False
            

    def _get_diff(self,X,y,w):
        cand_score = []

        for i in range(len(X)):
            cand_score.append(np.dot(np.abs(X[i]-y),w))
            
        return np.median(cand_score)
        
    
    def _update(self, times1, times2):


        file_name = str(int(time()))
        with open(self.file_path + file_name + '.csv', 'w', newline='') as csvfile:
            fieldnames = ['time1', 'time2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(times1)):
                
                
                writer.writerow({'time1': times1[i],
                                 'time2': times2[i]})
        '''
        f = open(self.file_path + file_name + '.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(times)
        wr.writerow(times2)
        f.close()
        '''
        #np.save(self.file_path+file_name,times)
        if self.debug:
            print('[debug] pattern save as',self.file_path+file_name)

        if self.n_patterns == self.n_current:
            patterns = glob(self.file_path+'*.csv')
            patterns.sort()
            if self.debug:
                print('[debug] remove ',patterns[0])    
            remove(patterns[0])
            
        
    def login(self):

        while True:
            
            records = recording(until='enter')
            sys.stdout.flush()
            y_time, y_name = self._record_to_time(records)
            y_time2 = self._record_to_time_another(records)
            
            y_time = np.array(y_time)
            
            #입력이 태그와 같은지 확인
            if self._typing_check(y_name):
                break
            else:
                print("[!] 입력이 일치하지 않습니다")
        
        #참조 패턴을 불러옴
        self.n_pattern = len(y_time)
        x_times, _ = self._load_data()
        #if self.debug:
            #print("[x times]")
            #print((1000*x_times).astype(np.int32))
        
        det = self._recognition(x_times, y_time)
        if det:
            print("login 성공")
            self._update(y_time, y_time2)

        else:
            print("login 실패")
        

if __name__ == '__main__':
    
    while True:
        k1 = KeyStroke('./number/',debug = True)
        k1.login()
        sys.stdout.flush()
        
        
        
