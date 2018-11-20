from lib.disk_struct import Disk
from algorithms.page_replacement_algorithm import  page_replacement_algorithm
from lib.priorityqueue import priorityqueue
from lib.CacheLinkedList import CacheLinkedList
import time
import numpy as np
import Queue
import heapq
# import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LeCaR_new(page_replacement_algorithm):

#     def __init__(self, N, visualization = True):
    def __init__(self, param):

        assert 'cache_size' in param

        self.N = int(param['cache_size'])
        self.H = int(self.N * int(param['history_size_multiple'])) if 'history_size_multiple' in param else self.N
        self.discount_rate = float(param['discount_rate']) if 'discount_rate' in param else 1
        self.learning_rate = float(param['learning_rate']) if 'learning_rate' in param else 0
        self.initial_weight = float(param['initial_weight']) if 'initial_weight' in param else 0.5
        self.Visualization = 'visualize' in param and bool(param['visualize'])
        self.discount_rate = 0.005 **(1/self.N)
        self.CacheRecency = CacheLinkedList(self.N)

        self.freq = {}
        self.PQ = []

        self.Hist1 = CacheLinkedList(self.H)
        self.Hist2 = CacheLinkedList(self.H)
        np.random.seed(123)

        ## Accounting variables
        self.time = 0
        self.eTime = {}

        self.timeIn = {} # Time page was inserted in cache
        self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)

        self.X = []
        self.Y1 = []
        self.Y2 = []

        self.unique = {}
        self.unique_cnt = 0
        self.pollution_dat_x = []
        self.pollution_dat_y = []
	self.pollution_dat_y_val = 0
        self.pollution_dat_y_sum = []

        self.info = {
                'lru_misses':0,
                'lfu_misses':0,
                'lru_count':0,
                'lfu_count':0,
            }

    def get_N(self) :
        return self.N

    def __contains__(self, q):
        return q in self.CacheRecency

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
        return np.array([self. X, self.Y1, self.Y2,self.pollution_dat_x,self.pollution_dat_y ]).T
#         return np.array([self.pollution_dat_x,self.pollution_dat_y ]).T

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
        self.timeIn[page] = self.time
        heapq.heappush(self.PQ, (self.freq[page], self.timeIn[page], page))

    ##########################################
    ## Add a page to cache using policy 'poly'
    ##########################################
    def addToCache(self, page):
        assert page not in self.CacheRecency, "Page already in cache"
        assert page not in self.timeIn, "Page already in cache"
        self.CacheRecency.add(page)
        # if page not in self.freq :
            # self.freq[page] = 0
        self.freq[page] = 1
        self.timeIn[page] = self.time
        # heapq.heappush(self.PQ, (self.freq[page], page))
        heapq.heappush(self.PQ, (self.freq[page], self.timeIn[page], page))

    def getHeapMin(self):
        # while self.PQ[0][1] not in self.CacheRecency or self.freq[self.PQ[0][1]] != self.PQ[0][0] :
        #     heapq.heappop(self.PQ)
        # return self.PQ[0][1]
        #self.print_state(self.PQ.getFreqDic().keys(), self.PQ.getFreqDic())

        while self.PQ[0][2] not in self.CacheRecency or self.freq[self.PQ[0][2]] != self.PQ[0][0] or self.timeIn[self.PQ[0][2]] != self.PQ[0][1]:
	    print "Entering = "
            heapq.heappop(self.PQ)
        return self.PQ[0][2]

    ######################
    ## Get LFU or LFU page
    ######################
    def selectEvictPage(self, policy):
        r = self.CacheRecency.getFront()
        f = self.getHeapMin()

        pageToEvit,policyUsed = None, None
        #if r == f :
            #pageToEvit,policyUsed = r,-1
        if policy == 0:
            pageToEvit,policyUsed = r,0
            #print "LRU ------------------------------"
        elif policy == 1:
            pageToEvit,policyUsed = f,1

        return pageToEvit,policyUsed

    def evictPage(self, pg):
        assert pg in self.CacheRecency
        assert pg in self.timeIn
        self.CacheRecency.delete(pg)
        del self.timeIn[pg]

    def getQ(self):
        lamb = 0.05
        return (1-lamb)*self.W + lamb
    ############################################
    ## Choose a page based on the q distribution
    ############################################
    def chooseRandom(self):
        r = np.random.rand()
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
            self.info['lru_count'] += 1
        else:
            if self.Hist2.size() == self.H  :
                histevict = self.Hist2.getFront()
                assert histevict in self.Hist2
                self.Hist2.delete(histevict)
            self.Hist2.add(cacheevict)
            self.info['lfu_count'] += 1

        if histevict is not None :
            del self.freq[histevict]
	    del self.eTime[histevict]

    ########################################################################################################################################
    ####REQUEST#############################################################################################################################
    ########################################################################################################################################
    def request(self,page) :
        page_fault = False
        self.time = self.time + 1

        #print "request = ", page
        #self.print_state(self.CacheRecency, self.freq)
	
	#print "PQ = ", self.PQ
	#print "heapq= ", heapq
        #self.print_state(self.PQ.getFreqDic().keys(), self.PQ.getFreqDic())
	
        ###########################
        ## Clean up
        ## In case PQ get too large
        ##########################
        if len(self.PQ) > self.N:
	    #print "entering cleaning"
            newpq = []
            for pg in self.CacheRecency:
                newpq.append((self.freq[pg], self.timeIn[pg], pg))
            heapq.heapify(newpq)
            self.PQ = newpq
            del newpq
	
	#print "PQ = ", self.PQ

        #####################
        ## Visualization data
        #####################
        if self.Visualization:
            self.X.append(self.time)
            self.Y1.append(self.W[0])
            self.Y2.append(self.W[1])


        ##########################
        ## Process page request
        ##########################
        if page in self.CacheRecency:
            page_fault = False
            self.pageHitUpdate(page)
        else :
            #####################################################
            ## Learning step: If there is a page fault in history
            #####################################################
            pageevict = None

            reward = np.array([0,0], dtype=np.float32)
            if page in self.Hist1:
                pageevict = page
                self.Hist1.delete(page)
                #reward[0] = -1
                reward[0] = -self.discount_rate **(self.time-self.eTime[page])
                self.info['lru_misses'] +=1

            elif page in self.Hist2:
                pageevict = page
                self.Hist2.delete(page)
                #reward[1] = -1
                reward[1] = -self.discount_rate **(self.time-self.eTime[page])
                self.info['lfu_misses'] +=1

            #################
            ## Update Weights
            #################
            if pageevict is not None  :
                self.W = self.W * np.exp(self.learning_rate * reward)
                self.W = self.W / np.sum(self.W)

            ####################
            ## Remove from Cache
            ####################
            if self.CacheRecency.size() == self.N:

                ################
                ## Choose Policy
                ################
                act = self.chooseRandom()
                cacheevict,poly = self.selectEvictPage(act)

                ###################
                ## Remove from Cache and Add to history
                ###################
                self.eTime[cacheevict] = self.time
                self.evictPage(cacheevict)
                self.addToHistory(poly, cacheevict)
                #print "evict = ", cacheevict

            self.addToCache(page)

            page_fault = True

        ## Count pollution


        if page_fault:
             self.unique_cnt += 1
        self.unique[page] = self.unique_cnt

        if self.time % self.N == 0:
            pollution = 0
            for pg in self.CacheRecency:
                if self.unique_cnt - self.unique[pg] >= 2*self.N:
                    pollution += 1

            self.pollution_dat_x.append(self.time)
            self.pollution_dat_y.append(100* pollution / self.N)
	self.pollution_dat_y_val  += 100* self.pollution / self.N
        self.pollution_dat_y_sum.append(self.pollution_dat_y_val)
        assert self.CacheRecency.size() <= self.N
        return page_fault
    
    def getPollutions(self):
        return self.pollution_dat_y_sum
    
    def getLearningRates(self):
        return self.learning_rates
    def get_list_labels(self) :
        return ['L']
