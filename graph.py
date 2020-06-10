import csv
import operator
import networkx as nx
import matplotlib.pyplot as plt 
import math

def findit(listit, val, left, right):
	if right<0:
		right = 0
	mid = (left+right)//2
	if left >= right:
		if left <= 0:
			return -1,right
		elif listit[left][0] == val:
			return left,right
		else:
			return -1,right
	if listit[mid][0] == val:
		extremel = mid
		extremer = mid
		for i in range(0, mid-left):
			if listit[mid-i][0] == val:
				extremel = extremel-1
			else:
				break
		for i in range(0, right-mid):
			if listit[mid+i][0] == val:
				extremer = extremer+1
			else:
				break
		return extremel,extremer
	elif listit[mid][0] > val:
		return findit(listit, val, left, mid-1)
	else:
		return findit(listit, val, mid+1, right)
def secondelement(listit, val, left, right):
	if right<0:
		right = 0
	mid = (left+right)//2
	if left >= right:
		if left <= 0:
			return -1,right
		elif listit[left][1] == val:
			return left,0
		else:
			return -1, right
	if listit[mid][1] == val:
		return mid,0
	elif listit[mid][1] > val:
		return findit(listit, val, left, mid)
	else:
		return findit(listit, val, mid, right)
def binfreq(listit, val, left, right):
	if right<0:
		right = 0
	mid = (left+right)//2
	if left >= right:
		if left <= 0:
			return -1,right
		elif listit[left][0] == val:
			return left,0
		else:
			return -1,right
	if listit[mid][0] == val:
		return mid,0
	elif listit[mid][0] > val:
		return findit(listit, val, left, mid)
	else:
		return findit(listit, val, mid, right)
edge = []
mostfreq = []
prev = 0
now = 0
G = nx.MultiDiGraph()

with open('preprocessed_400.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	accurate = 0.0
	tested = 0.0
	print ("predicted", " actual")
	k=0
	for row in reader:
		prev = now
		now = int(row[1])
		k=k+1
		left,right = findit(edge, prev, 0, len(edge)-1)
		if left == -1:
			edge.append([prev,now])
			edge = sorted(edge, key = lambda x: (x[0], x[1]))
			if k<1000:
				if prev == 0:
					G.add_edges_from([(0,math.log(now,5))])
				else:
					G.add_edges_from([(math.log(now,5),math.log(prev,5))])
			changefreq, upload = binfreq(mostfreq, prev, 0, len(mostfreq)-1)
			mostfreq.append([prev,now])
			mostfreq = sorted(mostfreq, key = lambda x: (x[0], x[1]))
			print ("cant predict")
		else:			
			leftsec, rightsec = secondelement(edge, now, left, right)
			changefreq, upload = binfreq(mostfreq, prev, 0, len(mostfreq)-1)
			print (mostfreq[changefreq][1]," ",now)
			tested = tested+1
			if mostfreq[changefreq][1] == now:
				accurate = accurate+1
			mostfreq[changefreq][1] = now
			if leftsec == -1:
				edge.append([prev,now])
				edge = sorted(edge, key = lambda x: (x[0], x[1]))
				if k<1000:
					if prev == 0:
						G.add_edges_from([(0,math.log(now,5))])
					else:
						G.add_edges_from([(math.log(now,5),math.log(prev,5))])
	print ("accuracy is:", (accurate/tested)*100,"%\n")
	print ("tested", tested)
	print ("accurate ", accurate)
#for plotting in edge:
#	if plotting[0] == 0:
#		G.add_edges_from([(0,math.log(plotting[1],5))])
#	else:
#		G.add_edges_from([(math.log(plotting[0],5),math.log(plotting[1],5))])
plt.figure(figsize=(100,100))
nx.draw(G, connectionstyle='arc3, rad = 0.0',node_size=20, arrowsize=10,)
plt.show()







