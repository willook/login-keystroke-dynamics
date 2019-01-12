import numpy as np
from glob import glob
from keyboard import record as recording
from time import time
from os import remove
from os import system
import sys
import csv
from key_graph import key_gragh

class KeyStroke():
    #file_path = None
    #tag_name = 'tag.txt'
    #n_patterns = 10
    #n_pattern = None
    #n_current = None
    
    def __init__(self,file_path, tag_name = 'tag.txt', n_patterns = 20, debug = False, update = True):
        self.file_path = file_path
        self.n_patterns = n_patterns
        self.n_current = None
        self.n_pattern = None
        self.n_valid = None
        self.tag_name = tag_name
        self.threshold = 1.25
        self.debug = debug
        self.update = update

        
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
        line = line.replace("\n","")
        return line == name

    def _load_data(self):
        
        patterns = glob(self.file_path+'*.csv')
        self.n_current = len(patterns)
        self.n_valid = int(self.n_current * 0.3)
        X = np.zeros((self.n_current, self.n_pattern))
        
        for i in range(len(patterns)):
            x = []
            
            with open(patterns[i], newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    x.append(row['times'])    
            X[i] = x
        return X

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
            weights[i] = 1/(np.var(X_T[i][:-self.n_valid])**0.5)*np.abs(np.mean(X_T[i]))
            #weights[i] = 1/(np.var(X_T[i][:-self.n_valid])**0.5)
        
        kg = key_gragh()
        f = open(self.file_path+self.tag_name)
        line = f.readline()
        line = line.replace("\n","")
        f.close()
        for i in range(len(line)-1):
            if kg.is_chain(line[i],line[i+1]):
                weights[i] = 0
        

        w_sum = np.sum(weights)
        for i in range(len(X_T)):
            weights[i] = weights[i]/w_sum
        return weights

    def _weights2(self,X):
        return np.ones(X[0].shape)/len(X[0])

    
        
    def _recognition(self, Xp, yp):

        yp = yp.astype(Xp.dtype)
        if self.n_current < self.n_patterns * 0.5:
            print("data가 충분치 않습니다")
            return True
        
        score = 0
        weights = self._weights(Xp)
        #weights = np.ones(yp.shape)/len(yp)

        
                
        
        if self.debug:
            print('[weight]')
            print((weights*100).astype(np.int32))

    
        
        for i in range(self.n_current - self.n_valid, self.n_current):
            med_score = self._get_diff(Xp[:-self.n_valid],Xp[i],weights)
            
            score += med_score / self.n_valid

     
        input_score = self._get_diff(Xp[:-self.n_valid],yp,weights)
        if self.debug:
            print("[input_score] :",input_score)
            print("[threshold] :",score * self.threshold)
        if input_score <= score * self.threshold:
            return True
        
        return False
            

    def _get_diff(self,X,y,w):
        cand_score = []
        #print(X.shape)
        #print(type(y))
        #print(int(y[0]*1000),int(y[-1]*1000))
        for i in range(len(X)):
            diff = np.abs(X[i]-y)
            cand_score.append(np.dot(diff,w))
        return np.median(cand_score)
        
    
    def _update(self, times):


        file_name = str(int(time()))
        with open(self.file_path + file_name + '.csv', 'w', newline='') as csvfile:
            fieldnames = ['times']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(times)):
                
                
                writer.writerow({'times': times[i]})
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

    def _totalTime(self,x):
        x = x.T
        return (x[-1] - x[0]).T

    def login(self, y = None):
        
        if y == None:
            while True:
                print("[id]: ")
                records = recording(until='enter')
                sys.stdout.flush()
                y_time, y_name = self._record_to_time(records)
                y_time2 = self._record_to_time_another(records)
                
                y_time = np.array(y_time)
                y_time2 = np.array(y_time2)
                
                #입력이 태그와 같은지 확인
                if self._typing_check(y_name):
                    break
                else:
                    print("[!] 입력이 일치하지 않습니다")
            yp = self._preprocess(y_time)
            yp2 = y_time2
            yp3 = self._totalTime(y_time)
            y = np.append(yp,yp2,axis=0)
            y = np.hstack([y,yp3])
        else:
            y = np.array(y)
                
        #참조 패턴을 불러옴
        self.n_pattern = len(y)
        X = self._load_data()

        #if self.debug:
            #print("[x times]")
            #print((1000*x_times).astype(np.int32))


        
        
 
        #from time import sleep
        #sleep(100)
        det = self._recognition(X, y)
        
        
        if det:
            print("login 성공")
            if self.update:
                self._update(y)
            return True

        else:
            print("login 실패")
            return False

def validation(train_path, test_paths,debug = True):
    k1 = KeyStroke(train_path,debug = True,n_patterns = 30,update = False)


    score = 0
    total = 0
    
    score1 = 0
    total1 = 0
    
    score2 = 0
    total2 = 0
    
    
    test_paths = glob(test_paths+"**/")
    print("test_paths[2]: ",len(test_paths))
    for test_path in test_paths:
        f = open(test_path + "answer.txt", 'r')
        label = bool(f.readline().replace('\n',''))
    
        f.close()
        
        csv_paths = glob(test_path+"*.csv")
        
        print("csv_path[s2x]: ",len(csv_paths))
        for csv_path in csv_paths:
           
            y = []
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    y.append(row['times'])   

            xlabel = k1.login(y)
            if label:
                if xlabel:
                    score+=1
                    score1+=1
                total1+=1
            else:
                if xlabel:
                    score2+=1
                else:
                    score+=1
                total2+=1  
            total += 1
            if debug:
                print("ans:",label, "guess:",xlabel)
                print(score, total)
                print()
    return score/total,score1/total1,score2/total2


if __name__ == '__main__':
    train = "./train/"
    test = "./test/"
    ret,ret1,ret2 = validation(train, test, debug = True)
    print("성공률:",ret)
    print("jh 성공률:",ret1)
    print("yt 성공률:",ret2)
    
    '''
    k1 = KeyStroke('./test/yt/',debug = True,n_patterns = 40,update = True)
    while True:
        k1.login()
        sys.stdout.flush()
    '''
