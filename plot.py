import numpy as np
import sys
import matplotlib.pyplot as plt

input_file = sys.argv[1]
f = open(input_file)

plt.title(input_file)
names = ['LRU', 'LFU', 'ARC']
hit_rates = [float(hr.strip()) for hr in next(f).split()]
#print(hit_rates)

lines = []
for line in f:
    if line.strip() == "":
        continue
    lines.append(line)
LeCaR = np.array( [[float(hr.strip()) for hr in line.split()]  for line in lines])
print(LeCaR)

x = LeCaR[:,0]
y = LeCaR[:,1]
l0,_ = plt.plot(x,y, 'k', x,y, 'ko', label="LeCaR")

lbl = [l0]
colors=['y', 'b', 'r']
for name, hit, color in zip(names, hit_rates, colors):
    l1 = plt.hlines(y=hit, xmin=x[0], xmax=x[-1], colors=color, label=name)
    lbl.append(l1)
plt.legend(handles=lbl, loc="upper right")

plt.xlabel('Learning Rate')
plt.ylabel('Hit Rate')
plt.show()
