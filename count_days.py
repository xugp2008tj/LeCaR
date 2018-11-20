import sys
import time

def nu_days(filename):
  filename = str(sys.argv[1])
  with open(filename) as f:
    count = 0
    writes = 0
    lines = 0
    x = 128166477394345573
    for line in f:
      line = line.split(',')
      count += 1
      if line[3] == 'Write': writes += 1
      if int(line[0]) - x <= 850833798273: 
        lines += 1
        #print(x - int(line[0]))
    print(count, writes, lines)

 
if __name__ == "__main__":
  nu_days(sys.argv[1])

