import numpy as np

class key_gragh():
    def __init__(self):
        self.di = [1,0,-1,1,0,-1,1,0,-1]
        self.dj = [1,1,1,0,0,0,-1,-1,-1]
        
        self.graph =[['1','2','3','4','5','6','7','8','9','0','-','='],
                     ['q','w','e','r','t','y','u','i','o','p','[',']'],
                     ['a','s','d','f','g','h','j','k','l',';',"'",],
                     ['z','x','c','v','b','n','m',',','.','/',]]
        

    def is_chain(self,a, b):
        

        
        for i in range(len(self.graph)):
            if a in self.graph[i]:
                j = self.graph[i].index(a)
                for k in range(len(self.di)):
                    ni = i+self.di[k]
                    nj = j+self.dj[k]
                
                    
                    if self._is_range(ni,nj) and self.graph[ni][nj] == b:
                         return True
        return False
        
    def _is_range(self, i,j):
        if i<0 or i>=len(self.graph):
            return False
        if j<0 or j>=len(self.graph[i]):
            return False
        return True
        
if __name__ == "__main__":
    k2 = key_gragh()
    print(k2.is_chain('1','2'))

