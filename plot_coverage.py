#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import matplotlib.pyplot as plt
import matplotlib.pyplot as plto
plt.xlabel("No. of deltas")
plt.ylabel("Coverage percentage")
plt.title("Delta Coverage Graph")
def findbin(listit,left,right,val):
  if left>=right:
    if right == -1:
      return -1
    if listit[left][0] == val:
      return left
    else:
      return -1
  mid = (left+right)//2
  if listit[mid][0] == val:
    return mid
  elif listit[mid][0]> val:
    return findbin(listit,left,mid-1,val)
  elif listit[mid][0]< val:
    return findbin(listit,mid+1,right,val)
  return -1

deltas = []
total = 0
inst = []
fileslist = ['510','511','526','600','602','620','623','625','631','641','648','657']
for j in range (0,13):
  print("-------------->", fileslist[j])
  for i in range (0,64):
    filename = '/nfs_home/nbhardwaj/data/rds_final/'+fileslist[j]+'_'+str(i)+'.csv'
    print ("total",total)
    with open(filename, 'rt') as f:
      reader = csv.reader(f, delimiter=',')
      print (filename)
      for row in reader:
        if row[0]=='':
          continue  
        total+=1
        if row[0] == '0':
          checkdata = 0
        else:
          checkdata = findbin(deltas,0,len(deltas)-1,int(float(row[7])))
          if checkdata == -1:
            deltas.append([int(float(row[7])),1])
            deltas.sort()
          else:
            deltas[checkdata][1]+=1
        checkins = findbin(inst,0,len(inst)-1,int(row[2]))
        if checkins == -1:
          inst.append([int(row[2]),1])
          inst.sort()
        else:
          inst[checkins][1]+=1
deltas.sort(key=lambda x: x[1], reverse= True)
inst.sort(key=lambda x: x[1], reverse= True)
data_covered = 0
data_coverage = []
for i in range (0,len(deltas)):
  if i>=50000:
    break
  data_covered+=deltas[i][1]
  data_coverage.append((data_covered/total)*100)
ins_covered = 0
instruction_coverage = []
for i in range (0,len(inst)):
  if i>=50000:
    break
  ins_covered+=inst[i][1]
  instruction_coverage.append((ins_covered/total)*100)
plt.plot(data_coverage, label = "Delta Coverage")
plt.legend()
plt.show()
plt.savefig('delta_coverage.png')
plto.xlabel("No. of instruction")
plto.ylabel("Coverage percentage")
plto.title("Instruction Coverage Graph")
plto.plot(instruction_coverage, label = "Instruction Coverage")
plto.legend()
plto.show()
plt.savefig('instruction_coverage.png')
print ("delta coverage = ",(data_covered/total)*100,"\n instruction coverage = ",(ins_covered/total)*100)

