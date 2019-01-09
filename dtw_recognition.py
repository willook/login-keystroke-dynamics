import librosa
import librosa.display
import numpy as np
from dtw import dtw
import sounddevice as sd
import time
from glob import glob



class recognition():       

    def loadData(self):
        with open('sounds3/tag2.txt') as f:
            self.labels = np.array([l.replace('\n', '') for l in f.readlines()])
        print(self.labels)
            
        #print("get labels complete")

        self.mfccs= {}
        for i in range(len(self.labels)):
            y, sr = librosa.load('sounds3/{}.wav'.format(i))
            #print(labels[i]) 
            #sd.play(y,sr)
            #time.sleep(2)
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)
            self.mfccs[i] = mfcc.T
            
        #self.mfccs = np.load("./sounds/mfccs.npy")
        #print(self.mfccs.shape)
        
        
    def recognition(self,x):
        
        dmin, jmin = np.inf, -1
        for i in range(len(self.mfccs)):
            
            y = self.mfccs[i]
            
            d, _, _, _ = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            if d < dmin:
                
                dmin = d
                imin = i
                #print(self.labels[imin],d,i)
            #print(self.labels[i],d)
        print(self.labels[imin])
        return self.labels[imin]
        
    def getMfcc(self,wav,sr):
        mfcc = librosa.feature.mfcc(wav, sr, n_mfcc=13)
        mfcc = mfcc.T
        return mfcc

class validation(recognition):
    def __init__(self, train_path, test_path, debug = False):
        self.loadData()
        
        self.test_paths = glob(test_path+"**/")
        self.debug = debug
        self.tag_name = "tag2.txt"

    def valid(self):
        score = 0
        total = 0

        for test_path in self.test_paths:
            f = open(test_path + self.tag_name, 'r')
            label = f.readline().replace('\n','')
            f.close()
            
            wav_paths = glob(test_path+"*.wav")
            for wav_path in wav_paths:
                print(wav_path)
                wav, sr = librosa.core.load(wav_path)
                x = self.getMfcc(wav,sr)
                xlabel = self.recognition(x)
                if xlabel == label:
                    score += 1
                total += 1
                if self.debug:
                    print("ans:",label, "guess:",xlabel)
                    print(score, total) 
        return score/total

if __name__ == '__main__':
    from time import time
    train_path = "./sound3/"
    test_path = "./test/"
    tic = time()
    v1 = validation(train_path, test_path, debug = True)
    ret = v1.valid()
    toc = time()
    print("recognition:",ret*100,"%")
    print("time:",toc-tic)


