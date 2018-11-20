import sys
import time
from algorithms.GetAlgorithm import GetAlgorithm
from lib.traces import Trace
import warnings
import itertools
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

##
## plot two fixed version vs adaptive for comparison
##

ANNOTATION_HEIGHT =0.7
IMAGE_FOLDER='output/'



def test_algorithm(algo, pages,  partition_size = 10) :
    hits = 0
    last_percent = -1
    num_pages = len(pages)

    partition_hit_rate = []
    hit_sum = []

    # print ''
    for i,p in enumerate(pages) :
        
        if p in algo :
            hits += 1
        
        algo.request(p)

        hit_sum.append(hits)

        ## Progres
        percent = int ((100.0 * (i+1) / num_pages))
        if percent != last_percent and percent % 10 == 0 :
            # print percent
            bars = int(percent / 10)
            sys.stdout.write('|')
            for i in range(bars) :
                sys.stdout.write('=')
            for i in range(10 - bars ) :
                sys.stdout.write(' ')
            sys.stdout.write('|\r')
            sys.stdout.flush()
            last_percent = percent

    for i in range(15 ) :
        sys.stdout.write(' ')
    sys.stdout.write('\r')
    sys.stdout.flush()
    return hits,partition_hit_rate,hit_sum

def get_algo_name(param):
    learning_rate = float(param['learning_rate']) if 'learning_rate' in param else 0
    if param['algorithm'].lower() == "lecar3" or param['algorithm'].lower() == "lecar4":
            algo_name= "LeCaR"
    elif param['algorithm'].lower() == "lecar8" or param['algorithm'].lower() == "lecar9":
            algo_name= "ALeCaR"
    else:
            algo_name= "ARC"
    return algo_name


def run(param,  ax_weight, ax_hitrate, exp_cnt):
    assert "input_data_location" in param, "Error: parameter 'input_data_location' was not found"
    assert "experiment_name" in param, "Error: parameter 'experiment_name' was not found"
    assert "cache_size" in param, "Error: parameter 'cache_size' was not found"
    assert "algorithm" in param, "Error: parameter 'algorithm' was not found"
    
    ###########################################################################
    ## Specify input folder
    ## Create a file input_data_location.txt and put in the config folder
    ###########################################################################
    DATA_FOLDER = param["input_data_location"]
    experiment_name = param['experiment_name']
        
    ###############################################################
    ## Read data
    ###############################################################
    trace_obj = Trace(512)
    trace_obj.read(DATA_FOLDER+experiment_name)
    pages = trace_obj.get_request()
    pages = pages[:int(param['trace_limit'])] if 'trace_limit' in param else pages
    
    num_pages = len(pages)
    unique_pages = trace_obj.unique_pages()
    cache_size_per = float(param['cache_size'])
    param['cache_size'] = int(round(unique_pages*cache_size_per)) if cache_size_per < 1 else int(cache_size_per)
    
    ###############################################################
    ## Simulate algorithm
    ###############################################################

    print(  "Experiment name:" , experiment_name.split("-")[0],  ", Cache size:", cache_size_per)
   

    algo = GetAlgorithm(param['algorithm'])(param)
        
    averaging_window_size = int(0.01*len(pages))
    start = time.time()
    hits, _, hit_sum = test_algorithm(algo, pages, partition_size=averaging_window_size)
    end = time.time()

    ###############################################################
    ## Visualize 
    ###############################################################
    visualize = 'visualize' in param and bool(param['visualize'])
    if visualize :
        algo_name =  get_algo_name(param)
        
        for v in trace_obj.vertical_lines :
            ax_hitrate.axvline(x=v,color='g',alpha=0.75)
            if param['algorithm'].lower() == "lecar8":
                ax_weight.axvline(x=v,color='g',alpha=0.75)
            
        
        
        temp = np.append(np.zeros(averaging_window_size), hit_sum[:-averaging_window_size])
        hitrate = (hit_sum-temp) / averaging_window_size
        
        ax_hitrate.set_xlim(0, len(hitrate))
        hitrate_plot = round(100.0 * hits / num_pages,2)
        colors =["red","green", "blue"]
        # ax_hitrate.set_title('LeCaR (Learning Rate vs Hit Rate)')
        if param['algorithm'].lower() == "lecar8":
            ax_hitrate.plot(range(len(hitrate)), hitrate,label=algo_name+ " - " + str(hitrate_plot), color= colors[exp_cnt%3] ,alpha=0.8)
        else:
            ax_hitrate.plot(range(len(hitrate)), hitrate,label=algo_name+ "(LR:" + param['learning_rate'] +") - " + str(hitrate_plot), color= colors[exp_cnt%3],alpha=0.8)
        
        if param['algorithm'].lower() == "lecar8":
        
            learnig_rates = algo.getLearningRates()
            print("Mean Learning Rate", np.mean(learnig_rates))
            print("Max Learning Rate", np.max(learnig_rates))
            print("Min Learning Rate", np.min(learnig_rates))
            ax_weight.set_ylabel('Learning Rate')
            ax_weight.plot(range(len(learnig_rates)), learnig_rates, 'r-', linewidth=3)
           
    
    del pages[:]
        
    return round(100.0 * hits / num_pages,2),  round(end-start,3)



def run_experiment(keys, values, exp_num = 1):
    fig = plt.figure(figsize=(10,6))
    
    ax_hitrate = plt.subplot(2,1,2)
    ax_weight = plt.subplot(2,1,1, sharex= ax_hitrate)
    
    exp_cnt =0
    algo_names = []
    
    for vals in itertools.product(*tuple(values)):
        param = {}
        parameters = "" 
        for k, v in zip(keys, vals) :
            parameters += "{:<20}".format(v[-20:].strip())
            param[k] = v
        cache_size_per = float(param['cache_size'])
        algo_name= get_algo_name(param)
        if "ALeCaR" not in algo_names and algo_name == "ALeCaR" :
            algo_names.append(algo_name)
            hit_rate, duration = run(param, ax_weight, ax_hitrate, exp_cnt)
        elif algo_name != "ALeCaR" :
            algo_names.append(algo_name)
            hit_rate, duration = run(param, ax_weight, ax_hitrate, exp_cnt)
        exp_cnt+=1
        parameters += "{:<20}".format(hit_rate)
        parameters += " : {:<10}".format(duration)
        
        print(parameters)
    

    ax_weight.set_ylim(-0.05,0.3)
    ax_weight.set_ylabel('Weight')
    ax_weight.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax_hitrate.set_ylim(-0.05,0.3)        
    ax_hitrate.set_xlabel('Requests')
    ax_hitrate.set_ylabel('Hit Rate')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    ax_hitrate.legend(fancybox=True, framealpha=0.5)
    
    plt.savefig("output/%s_%s_%s_%s.png" % (param['experiment_name'].split("-")[0],str(cache_size_per),param['cache_size'], param['learning_rate']))
    
    plt.clf()

if __name__ == "__main__" :
    
    for config_idx in range(1, len(sys.argv)):
    
        config_file = open(sys.argv[config_idx], 'r')
        
        
        keys = []
        values = []
        header = ""
        exp_cnt = 1
        for line in config_file:
            if line.strip() == "":
                header += "{:<20}".format("hit rate")
                print(header)
                run_experiment(keys, values, exp_cnt)
                exp_cnt += 1
                del keys[:]
                del values[:]
                header = ""
                print("\n\n")
                continue
            key, vals = line.strip().split(":")
            keys.append(key)
            values.append(vals.strip().split(","))
            header += "{:<20}".format(key[-18:])
        
        if len(values)>0:
            header += "{:<20}".format("hit rate")
            print(header)
            run_experiment(keys, values, exp_cnt)
        
                
#         print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format("Name","Hit Ratio(%)", "Hit Count", "Total Request","Unique Pages", "Time") )
#         print("\n")
