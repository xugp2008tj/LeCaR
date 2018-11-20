import matplotlib as plt
import numpy as np

hit_sum= [0,1,1,2,3,4,5,6,6,7,8]
averaging_window_size= 3

temp = np.append(np.zeros(averaging_window_size), hit_sum[:-averaging_window_size])
print(temp)
hitrate = (hit_sum-temp) / averaging_window_size
print(hit_sum - temp)
print(hitrate)
        
# ax_hitrate.set_xlim(0, len(hitrate))