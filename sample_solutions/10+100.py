
# solution to task 2 of sheet 1 by Torge Wolff

import numpy as np

bitstring_len = 10
cnt_parents = 10
sigma = 1./bitstring_len
cnt_generations = 100
iters = 100

def _fitness(childs, final):
	
	fit = np.sum(childs, axis=1)

	if final:
		print("OneMaxFitness: ")
		print(fit)
		print("Mittelwert: ")
		print(np.mean(fit))
		print("Standardabweichung: ")
		print(np.std(fit))
		print("Median: ")
		print(np.median(fit))
		print("Min OneMax: ")
		print(np.min(fit))
		print("Max OneMax: ")
		print(np.max(fit))

	return fit

	
def _init_population(n):
	x = [0] * bitstring_len

	pars = [x for i in range(n)]

	return pars


def _two_point_crossover(parents):

	childs = []

	do = True

	while(do):
		rand_1 = np.random.randint(low=0, high=cnt_parents)
		rand_2 = np.random.randint(low=0, high=cnt_parents)
		
		if (rand_1 != rand_2):	

			parent_a = parents[rand_1]
			parent_b = parents[rand_2]

			low = np.random.randint(low=0, high=bitstring_len-1)
			high = np.random.randint(low=low+1, high=bitstring_len)

			assert low < high and low != high

			child_1 = [parent_a[i] if i <= low or i > high else parent_b[i] for i in range(bitstring_len)]
			child_2 = [parent_a[i] if i > low and i <= high else parent_b[i] for i in range(bitstring_len)]
			
			childs.append(child_1)
			childs.append(child_2)

		if len(childs) >= 100:
			do = False

	return childs


def _mutation(childs):

	childs_mutated = [[(bit + 1 ) % 2 if np.random.random() < sigma else bit for bit in child] for child in childs]

	return childs_mutated


def _selection(childs, fitness):

	assert len(childs) == len(fitness)
	
	mapped_childs = list(zip(fitness, childs)) 

	sorted_maped_childs = sorted(mapped_childs, reverse=True)
	top_childs = sorted_maped_childs[:cnt_parents]

	return_childs = [child[1] for child in top_childs]

	return return_childs


for k in range(iters):

	# create new initial population
	parents = _init_population(cnt_parents)

	for i in range(cnt_generations):

		# crossover with parents
		childs = _two_point_crossover(parents)
		
		# mutate new childs
		childs_mutated = _mutation(childs)

		#calculate fitness 
		fitness = _fitness(childs_mutated, False)

		# termination condition
		if np.max(fitness) == bitstring_len:
			break

		# select new parents for next generation
		parents = _selection(childs_mutated, fitness)

	print("Fitness of %i:" % k)
	print(_fitness(parents, True))
	print("--" * 10)

