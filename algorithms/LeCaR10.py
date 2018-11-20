from lib.disk_struct import Disk
from algorithms.page_replacement_algorithm import  page_replacement_algorithm
from lib.priorityqueue import priorityqueue
from lib.CacheLinkedList import CacheLinkedList
import time
import numpy as np
import Queue
import heapq
import Queue as queue
# import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# sys.path.append(os.path.abspath("/home/giuseppe/))

## Keep a LRU list.
## Page hits:
##      Every time we get a page hit, mark the page and also move it to the MRU position
## Page faults:
##      Evict an unmark page with the probability proportional to its position in the LRU list.
class LeCaR10(page_replacement_algorithm):

#     def __init__(self, N, visualization = True):
    def __init__(self, param):

        assert 'cache_size' in param
        # assert 'history_size_multiple' in param

        self.N = int(param['cache_size'])
        self.H = int(self.N * int(param['history_size_multiple'])) if 'history_size_multiple' in param else self.N
        self.learning_rate = float(param['learning_rate']) if 'learning_rate' in param else 0
        
        self.discount_rate = float(param['discount_rate']) if 'discount_rate' in param else 1
        # self.discount_rate = (float(param['discount_rate']) if 'discount_rate' in param else 0) ** (1/self.N)
        # self.discount_rate = 0.05**(1/self.N)
        
        self.Visualization = 'visualize' in param and bool(param['visualize'])
        self.lamb = 0.05

        self.learning_rate = 0.1
        self.learning_rate_lfu= 0.1
        self.learning_rate_lru= 0.1



        self.CacheRecency = CacheLinkedList(self.N)


        self.freq = {}
        self.PQ = []

        self.Hist1 = CacheLinkedList(self.H)
        self.Hist2 = CacheLinkedList(self.H)
        np.random.seed(123)
        
        self.PageCount = 0
        self.CacheHit = 0
        
        self.PreviousHR = 0.0
        self.NewHR = 0.0
        self.PreviousChangeInHR = 0.0
        self.NewChangeInHR =0.0
        self.PreviousLR= 0.45
        self.NewLR =0.45
        self.CacheHitList = []
        self.counter = 0
        self.learning_rates = []
        self.SampleHR =[]
        self.SAMPLE_SIZE = 20 * self.N
        self.SampleHitQ = queue.Queue(maxsize= self.SAMPLE_SIZE)
        self.SampleCacheHit = 0
        

       


        ## Accounting variables
        self.time = 0
        self.W = np.array([.5,.5], dtype=np.float32)
        self.PreviousW = np.array([.5,.5], dtype=np.float32)
        self.NewW = np.array([.5,.5], dtype=np.float32)
        self.qUsed = {}
        self.eTime = {}


        self.X = []
        self.Y1 = []
        self.Y2 = []

        self.unique = {}
        self.unique_cnt = 0
        self.pollution_dat_x = []
        self.pollution_dat_y = []
        self.pollution_dat_y_val = 0
        self.pollution_dat_y_sum = []
        self.pollution =0

    def __contains__(self, q):
        return q in self.CacheRecency

    def get_N(self) :
        return self.N

    def visualize(self, ax_w, ax_h, averaging_window_size):
        lbl = []
        if self.Visualization:
            X = np.array(self.X)
            Y1 = np.array(self.Y1)
            Y2 = np.array(self.Y2)
            ax_w.set_xlim(np.min(X), np.max(X))
            ax_h.set_xlim(np.min(X), np.max(X))

            ax_w.plot(X,Y1, 'y-', label='W_lru', linewidth=2)
            ax_w.plot(X,Y2, 'b-', label='W_lfu', linewidth=1)
            #ax_h.plot(self.pollution_dat_x,self.pollution_dat_y, 'g-', label='hoarding',linewidth=3)
	         #ax_h.plot(self.pollution_dat_x,self.pollution_dat_y, 'k-', linewidth=3)
            ax_h.set_ylabel('Hoarding')
            ax_w.legend(loc=" upper right")
            ax_w.set_title('LeCaR - Adaptive LR')
            pollution_sums = self.getPollutions()
            temp = np.append(np.zeros(averaging_window_size), pollution_sums[:-averaging_window_size])
            pollutionrate = (pollution_sums-temp) / averaging_window_size
        
            ax_h.set_xlim(0, len(pollutionrate))
        
            ax_h.plot(range(len(pollutionrate)), pollutionrate, 'k-', linewidth=3)



#             lbl.append(l1)
#             lbl.append(l2)
#             lbl.append(l3)

        return lbl

    def getWeights(self):
        return np.array([self. X, self.Y1, self.Y2,self.pollution_dat_x,self.pollution_dat_y_sum ]).T
#         return np.array([self.pollution_dat_x,self.pollution_dat_y ]).T
    
    def getPollutions(self):
        return self.pollution_dat_y_sum
    
    def getLearningRates(self):
        return self.learning_rates

    def getStats(self):
        d={}
        d['weights'] = np.array([self. X, self.Y1, self.Y2]).T
        d['pollution'] = np.array([self.pollution_dat_x, self.pollution_dat_y ]).T
        return d

    ##############################################################
    ## There was a page hit to 'page'. Update the data structures
    ##############################################################
    def pageHitUpdate(self, page):
        assert page in self.CacheRecency and page in self.freq
        self.CacheRecency.moveBack(page)
        self.freq[page] += 1
        heapq.heappush(self.PQ, (self.freq[page],page))

    ##########################################
    ## Add a page to cache using policy 'poly'
    ##########################################
    def addToCache(self, page):
        self.CacheRecency.add(page)
        if page not in self.freq :
            self.freq[page] = 0
        self.freq[page] += 1
        heapq.heappush(self.PQ, (self.freq[page],page))

    def getHeapMin(self):
        while self.PQ[0][1] not in self.CacheRecency or self.freq[self.PQ[0][1]] != self.PQ[0][0] :
            heapq.heappop(self.PQ)
        return self.PQ[0][1]

    ######################
    ## Get LFU or LFU page
    ######################
    def selectEvictPage(self, policy):
        r = self.CacheRecency.getFront()
        f = self.getHeapMin()

        pageToEvit,policyUsed = None, None
        if r == f :
            pageToEvit,policyUsed = r,-1
        elif policy == 0:
            pageToEvit,policyUsed = r,0
        elif policy == 1:
            pageToEvit,policyUsed = f,1

        return pageToEvit, policyUsed

    def evictPage(self, pg):
        assert pg in self.CacheRecency
        self.CacheRecency.delete(pg)


    def getQ(self):
        lamb = 0.05
        return (1-lamb)*self.W + lamb/2
    ############################################
    ## Choose a page based on the q distribution
    ############################################
    def chooseRandom(self):
        # np.random.seed(10)
        r = np.random.rand()
#         q = self.getQ()
        if r < self.W[0] :
            return 0
        return 1

    def addToHistory(self, poly, cacheevict):
        histevict = None
        if (poly == 0) or (poly==-1 and np.random.rand() <0.5):
            if self.Hist1.size() == self.H  :
                histevict = self.Hist1.getFront()
                assert histevict in self.Hist1
                self.Hist1.delete(histevict)
            self.Hist1.add(cacheevict)
        else:
            if self.Hist2.size() == self.H  :
                histevict = self.Hist2.getFront()
                assert histevict in self.Hist2
                self.Hist2.delete(histevict)
            self.Hist2.add(cacheevict)

        if histevict is not None :
            del self.freq[histevict]
            del self.qUsed[histevict]
            del self.eTime[histevict]

    def updateLearningRates(self,seq_len):
        if self.time % (seq_len) == 0:
                
                # print("Inside",self.N/2)
                # self.NewHR = np.mean(self.CacheHitList)
                self.NewHR = self.CacheHit/ float(seq_len)
                
                self.NewChangeInHR= (self.NewHR -self.PreviousHR)
                self.NewCahngeInW = self.W - self.PreviousW
                    
                delta_1 = self.NewChangeInHR 
                delta_2 = self.PreviousChangeInHR
                delta = delta_1 * delta_2
                delta_lfu = 0
                delta_lru  = 0
                # print( "ChangeinW", abs(self.NewCahngeInW[0]), abs(self.NewCahngeInW[1]))
                if self.NewCahngeInW[0] != 0:
                    delta_lfu = self.NewChangeInHR /  (self.NewCahngeInW[0])
                if self.NewCahngeInW[1] != 0:
                    delta_lru = self.NewChangeInHR /  (self.NewCahngeInW[1])
                # print("Change in HR",self.NewChangeInHR )
                
                # print("LFU Previous and New Weights",self.PreviousW[0], self.W[0])
                # print("LRU Previous and New Weights",self.PreviousW[1], self.W[1])

                # print("delta", delta_lfu, delta_lru)

               

                # if self.learning_rate * (self.SampleCacheHit/ float(self.SAMPLE_SIZE)) <10**(-5):
                #     # if self.learning_rate==0 :
                #         self.learning_rate = 0.1 
                    # else:  self.learning_rate = self.learning_rate*2
                # elif self.learning_rate * (self.SampleCacheHit/ float(self.SAMPLE_SIZE)) <10**(-6):
                # # if self.learning_rate * delta * delta_1 <10**(-5):
                #         self.learning_rate =  0.45
                
                    
                if delta_lfu< 0 and delta_lfu!=0:
                       
                    self.learning_rate_lfu = max(self.learning_rate_lfu /2, 0  )
                elif  delta_lfu >0 and delta_lfu!=0:   
                    self.learning_rate_lfu = min(self.learning_rate_lfu + (0.1* self.learning_rate_lfu), 1  )
                
                if delta_lru< 0 and delta_lru!=0:
                       
                    self.learning_rate_lru = max(self.learning_rate_lru/2 , 0  )
                elif  delta_lru >0 and delta_lru!=0:  
                    self.learning_rate_lru = min(self.learning_rate_lru + (0.1* self.learning_rate_lru), 1  )
                
                  
               
                self.PreviousLR = self.NewLR

                self.NewLR = self.learning_rate
                

                self.PreviousHR = self.NewHR
                
                self.PreviousChangeInHR = self.NewChangeInHR
                del self.CacheHitList[:]
                self.CacheHit = 0
                self.PreviousW = self.W
                
               

              
          


    ########################################################################################################################################
    ####REQUEST#############################################################################################################################
    ########################################################################################################################################
    def request(self,page) :
        page_fault = False
        self.time = self.time + 1

       
        

        # print(self.PageCount)

        ###########################
        ## Clean up
        ## In case PQ get too large
        ##########################
        if len(self.PQ) > 2*self.N:
            newpq = []
            for pg in self.CacheRecency:
                newpq.append((self.freq[pg],pg))
            heapq.heapify(newpq)
            self.PQ = newpq
            del newpq

        #####################
        ## Visualization data
        #####################
        if self.Visualization:
            self.X.append(self.time)
            self.Y1.append(self.W[0])
            self.Y2.append(self.W[1])

        
       

        #####################################################
        ## Adapt learning rate Here
        ###################################################### 
        seq_len = self.N
        self.updateLearningRates(seq_len)
        # if len(self.SampleHR)== 20* self.N:
        #     del self.SampleHR[0]
        if self.SampleHitQ.full():
            self.SampleCacheHit -= self.SampleHitQ.get()
        
         ##########################
        ## Process page request
        ##########################
       


        
        
        if page in self.CacheRecency:
            page_fault = False
            self.CacheHit +=1
            self.CacheHitList.append(1)
            self.SampleCacheHit += 1
            self.SampleHitQ.put(1)
            # self.SampleHR.append(1)            
            self.pageHitUpdate(page)
        
        else :
            # updateLearningRates(self,seq_len)

            #####################################################
            ## Learning step: If there is a page fault in history
            #####################################################
            pageevict = None
            self.CacheHitList.append(0)
            self.SampleHitQ.put(0)
            # self.SampleHR.append(0)    

            reward = np.array([0,0], dtype=np.float32)
            if page in self.Hist1:
                pageevict = page
                self.Hist1.delete(page)
                
                # reward[0] = -1 / self.qUsed[page] ## punish
                reward[0] = -self.discount_rate **(  (self.time-self.eTime[page])  )  ## punish

                # reward[0] = -1 ## punish

            elif page in self.Hist2:
                pageevict = page
                self.Hist2.delete(page)
                # reward[1] = -1 / self.qUsed[page]
                reward[1] = -self.discount_rate ** (  (self.time-self.eTime[page])  ) 

                # reward[1] = -1 ## punish
            

            #################
            ## Update Weights
            #################
            if pageevict is not None  :
                # self.W = self.W * np.exp(self.learning_rate * reward )
                self.W[0] = self.W[0] * np.exp(self.learning_rate_lfu * reward[0] )
                self.W[1] = self.W[1] * np.exp(self.learning_rate_lru * reward[1] )
                self.W = self.W / np.sum(self.W)
                # self.W[0] = min(1-self.lamb, self.W[0])
                # self.W[0] = max(self.lamb, self.W[0])
                # self.W[1] = 1 - self.W[0]
                # self.PreviousW = self.W

            ####################
            ## Remove from Cache
            ####################
            if self.CacheRecency.size() == self.N:

                ################
                ## Choose Policy
                ################
                act = self.chooseRandom()
                cacheevict,poly = self.selectEvictPage(act)
#                 self.qUsed[cacheevict] = self.getQ()[poly]
                self.qUsed[cacheevict] = self.W[poly]

                self.eTime[cacheevict] = self.time
                ###################
                ## Remove from Cache and Add to history
                ###################
                self.evictPage(cacheevict)
                self.addToHistory(poly, cacheevict)

            self.addToCache(page)

            page_fault = True
         ## Count pollution


        if page_fault:
            self.unique_cnt += 1
        self.unique[page] = self.unique_cnt

        if self.time % self.N == 0:
            self.pollution = 0
            for pg in self.CacheRecency:
                if self.unique_cnt - self.unique[pg] >= 2*self.N:
                    self.pollution += 1

            self.pollution_dat_x.append(self.time)
            self.pollution_dat_y.append(100* self.pollution / self.N)
        self.pollution_dat_y_val  += 100* self.pollution / self.N
        self.pollution_dat_y_sum.append(self.pollution_dat_y_val)

        self.learning_rates.append(self.learning_rate)
        return page_fault

    def get_list_labels(self) :
        return ['L']

