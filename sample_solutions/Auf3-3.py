import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN

counter = 100
sigma = 1.
N = 2



def getKey(item):
    return item[1]


#def Sphere(solution):
#    return np.dot(solution,solution)

def Schwefel(solution):
	x = [0] * (len(solution))
	for i in xrange(N):
		x[i] = solution[i] * math.sin(math.sqrt(math.fabs(solution[i])))
	return 418.9829 * N - sum(x)
	

    
print "Generating Solutions :"
    
Solutions = []
Fitnesses = []
for i in xrange(100):
    solution = np.random.normal(100,200,N)
    #solution = np.random.uniform(0,1,N)
    #why it dosent work
    Solutions.append(solution)
    Fitnesses.append(Schwefel(solution))

zipped = zip(Solutions,Fitnesses)
zipped =  sorted(zipped, key=getKey)

Solutions= [i[0] for i in zipped]
Solutions_x_axis = [i[0] for i in Solutions]
Solutions_y_axis = [i[1] for i in Solutions]

print zipped

db = DBSCAN(eps=0.3,metric = 'euclidean' , min_samples=3).fit(Solutions)
labels = db.labels_
clusters_labels = set(labels)

print labels
print "the clusters are :" ,clusters_labels

optimal_solutions_in_all_clusters = []
for c in clusters_labels:

	for i in xrange(len(Solutions)):
		if labels[i] == c:
			choosen_solution = Solutions[i]
			break

	solution = choosen_solution
	print "we found a solution :", choosen_solution ," from cluster :" ,c
	print "Rnning 1+1 GA inside cluster :",c ,"using :" ,choosen_solution ," as an initial population. "
	for i in range(0,counter):
		sigma *=0.99
	
		print "Generation : ",i," fitness :",Schwefel(solution)
		solution_mutated = solution + sigma * np.random.standard_normal(N)
	
		if Schwefel(solution_mutated) < Schwefel(solution):
			solution = solution_mutated
	optimal_solutions_in_all_clusters.append(solution)
	print "the local optimal solution in cluster :",c ," is :",solution


colors = np.array([i for i in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
plt.scatter(Solutions_x_axis,Solutions_y_axis, color=colors[labels].tolist(), s=10)

for i in optimal_solutions_in_all_clusters:
	plt.scatter(i[0],i[1],c='g',s=100)
	print i



plt.show()


