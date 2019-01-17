import numpy as np
from glob import glob
from keyboard import record as recording
from time import time
from os import remove
from os import system
import sys
import csv
from key_graph import key_gragh
from sklearn import mixture
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
class KeyStroke():
    #file_path = None
    #tag_name = 'tag.txt'
    #n_patterns = 10
    #n_pattern = None
    #n_current = None
    
    def __init__(self,file_path,threshold = 1.50, GMM = False , n_patterns = 20, debug = False, update = True):
        self.file_path = file_path
        self.n_patterns = n_patterns
        self.n_current = None
        self.n_pattern = None
        self.n_valid = None
        self.tag_name = 'tag.txt'
        self.threshold = threshold
        self.debug = debug
        self.update = update
        self.GMM = GMM

        
    def _record_to_time(self, records):
        names = []
        times = []

        if records[0].name == 'enter' and records[0].event_type == 'up':
            records = records[1:]
        
        t0 = records[0].time

        self._delete = []
        #마지막 인자는 엔터이므로 체크하지 않음
        for i in range(len(records)):
            if records[i].event_type == 'down':
                if records[i].name == 'space':
                    records[i].name = ' '
                if records[i].name == 'enter':
                    records[i].name = ''
                
                if records[i].name == 'shift':
                    if i != 0 and records[i-1].name == 'shift':
                        times = times[:-1]
                        continue
                    else:
                        records[i].name = ''
                
                if records[i].name == 'backspace' or records[i].name == 'delete':
                    times = times[:-1]
                    names = names[:-1]
                    if i == 0:
                        self._delete = np.append(self._delete, i)
                    else:
                        self._delete = np.append(self._delete,i-2)
                        self._delete = np.append(self._delete,i-1)
                        self._delete = np.append(self._delete,i)
                        
                          
                else:
                    times.append(records[i].time-t0)
                    names.append(records[i].name)

        name = ''.join(names)
        
        return np.array(times), name
    
    def _record_to_time_another(self, records):
        #records = np.delete(records,self._delete,0)
        times_2 = []

        for i in range(len(records)-1):
            for j in range(i + 1, len(records) - 1):
                if records[i].event_type == 'down':
                    if records[i].name == 'backspace' or records[i].name == 'delete':
                        times_2 = times_2[:-1]            
                        break

                    
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

        return np.array(times_2)
    
    def _typing_check(self, name, pw = False):
        f = open(self.file_path+self.tag_name)
        if pw:
            f.readline()
        line = f.readline()     
        line = line.replace("\n","")
        if not pw:
            print("id: ",name)
        if pw:
            print("pw: ",name)
        
        f.close()
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
        #X = X[:,:2]          
        return X

    def plot_results(self,X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        #plt.xlim(-9., 5.)
        #plt.ylim(-3., 6.)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
        plt.show()

    

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
        weights = np.ones(X[0].shape)/len(X[0])*100
        
        kg = key_gragh()
        f = open(self.file_path+self.tag_name)
        line = f.readline()
        line = line.replace("\n","")
        f.close()
        for i in range(len(line)-1):
            if kg.is_chain(line[i],line[i+1]):
                weights[i] = 0
    
        return weights


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

    def _recognition_GMM(self, Xp, yp):

        yp = yp.astype(Xp.dtype)
        if self.n_current < self.n_patterns * 0.5:
            print("data가 충분치 않습니다")
            return True

        
        kg = key_gragh()
        f = open(self.file_path+self.tag_name)
        line = f.readline()
        line = line.replace("\n","")
        f.close()
        deleted = []
        for i in range(len(line)-1):
            if kg.is_chain(line[i],line[i+1]):
                deleted = np.append(deleted,i)
                deleted = np.append(deleted,int(i+len(yp)/2-1))
                
        Xp = np.delete(Xp,deleted,1)
        yp = np.delete(yp,deleted,0)
        
        #weights = self._weights(Xp)
        #self.gmm = mixture.GaussianMixture(n_components=1, covariance_type='spherical').fit(Xp)
        self.gmm = mixture.BayesianGaussianMixture(n_components=3, covariance_type='full').fit(Xp)
        #self.plot_results(Xp, self.gmm.predict(Xp), self.gmm.means_, self.gmm.covariances_, 0,'Gaussian Mixture')
        yp_T = np.reshape(yp,(1,len(yp)))
        
        input_score = self.gmm.score(yp_T)
        if self.debug:
            print(input_score)
        #self.threshold = -33.5  #bayesian n_component = 1, yp1 
        
        if self.debug:
            print("[input_score] :",input_score)
            print("[threshold] :",self.threshold)
        if input_score >= self.threshold:
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
        return np.min(cand_score)
        
    
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
        #x = x.T
        return x[-1] - x[0]

    def login(self, y = None):
        
        if y == None:
            while True:
                print("[id]: ")
                id_records = recording(until='enter')
                from time import sleep
                sleep(0.5)

                print("[pw]: ")
                pw_records = recording(until='enter')
                sleep(0.5)
                #sys.stdout.flush()
                id_y_time, id_y_name = self._record_to_time(id_records)
                id_y_time2 = self._record_to_time_another(id_records)

                pw_y_time, pw_y_name = self._record_to_time(pw_records)
            
                pw_y_time2 = self._record_to_time_another(pw_records)
                
                #print(y_name)
                #입력이 태그와 같은지 확인
                if self._typing_check(id_y_name) and self._typing_check(pw_y_name, pw=True):
                    break
                else:
                    print("[!] id나 비밀번호가 일치하지 않습니다.")
            
            id_yp = self._preprocess(id_y_time)
            pw_yp = self._preprocess(pw_y_time)
            yp = np.append(id_yp, pw_yp,axis=0)
            
            yp2 = np.append(id_y_time2, pw_y_time2,axis=0)
            id_yp3 = self._totalTime(id_y_time)
            pw_yp3 = self._totalTime(pw_y_time)
            
            y = np.append(yp,yp2,axis=0)
            y = np.hstack([y,id_yp3])
            y = np.hstack([y,pw_yp3])
            
        else:
            y = np.array(y,dtype = np.float64)
                    
        #참조 패턴을 불러옴
        self.n_pattern = len(y)
        X = self._load_data()

        det = None
        if self.GMM:
            det = self._recognition_GMM(X, y)
        else:
            det = self._recognition(X, y)
            
        
        if det:
            if self.debug:
                print("login 성공")
            if self.update:
                self._update(y)
            return True

        else:
            if self.debug:
                print("login 실패")
            return False

def validation(train_path, test_paths,debug = True, threshold=1.5,GMM = False):
    k1 = KeyStroke(train_path,threshold=threshold,debug = debug, n_patterns = 20,update = False,GMM=GMM)


    score = 0
    total = 0
    
    score1 = 0
    total1 = 0
    
    score2 = 0
    total2 = 0
    
    
    test_paths = glob(test_paths+"**/")
    for test_path in test_paths:
        f = open(test_path + "answer.txt", 'r')
        label = bool(f.readline().replace('\n',''))
    
        f.close()
        
        csv_paths = glob(test_path+"*.csv")
        
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

    train = "./train3/"
    test = "./test3/"
    
    #threshold of gmm -33.5 = 79%
    #threshold of nomal 1.5 = 79%
    '''
    ret,ret1,ret2 = validation(train, test,threshold=1.5, debug = True, GMM=False)
    print("성공률:",ret)
    print("본인 성공률:",ret1)
    print("타인 성공률:",ret2)
    '''
    '''
    #min_thres = 1.40
    #max_thres = 1.80
    min_thres = -8000
    max_thres = -8000
    n=1
    for i in range(n):
        threshold = min_thres + (max_thres-min_thres)*i/n
        ret,ret1,ret2 = validation(train, test,threshold=threshold,GMM=True, debug = False)
        print("성공률:",ret,threshold)
        print("jh 성공률:",ret1)
        print("yt 성공률:",ret2)
    '''
    #train = "./test2/yt/"
    k1 = KeyStroke(train, debug = True,threshold = 1.5,n_patterns = 10,update = True,GMM=False)
    while True:
        k1.login()
        sys.stdout.flush()
    
