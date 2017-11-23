import numpy as np
import math
import matplotlib.pyplot as plt

populationCount = 1
population = [0] * populationCount
offspringCount = 1
offspring = [0] * offspringCount

tau = [1.2, 1.5, 1.8, 5]
tauIndicator = 0
successCount = 0

N = [2, 10, 100]
NIndicator = 0

generations = [100, 1000]
generationsIndicator = 0
generationsCounter = 1

sigma = None
lastFitness = None


def crossover():
	global offspring


# nothing to todo as 1+1 ga cannot have crossover

def selection():
	global offspring
	global population


# nothing todo as 1+1 ga does not need a selection

def init():
	global population
	global lastFitness
	global sigma
	mean = 100
	standardDeviation = mean * 0.5
	# print np.random.normal(mean, standardDeviation, N[NIndicator])
	population = [np.random.normal(mean, standardDeviation, N[NIndicator]) for individual in population]
	lastFitness = worstFitnesses[functionIndicator]
	sigma = 0.5


def sphere(individual):
	x = np.power(individual, 2)
	return sum(x)


def rosenbrock(individual):
	x = [0] * (len(individual) - 1)
	for i in xrange(N[NIndicator] - 1):
		x[i] = 100 * math.pow(individual[i + 1] - math.pow(individual[i], 2), 2) + math.pow(individual[i] - 1, 2)
	return sum(x)


def rastrigin(individual):
	x = [0] * (len(individual))
	for i in xrange(N[NIndicator]):
		x[i] = math.pow(individual[i], 2) - 10 * math.cos(2 * math.pi * individual[i])
	return 10 * N[NIndicator] + sum(x)


def schwefel(individual):
	x = [0] * (len(individual))
	for i in xrange(N[NIndicator]):
		x[i] = individual[i] * math.sin(math.sqrt(math.fabs(individual[i])))
	return 418.9829 * N[NIndicator] + sum(x)


def fitness():
	functions = [sphere, rosenbrock, rastrigin, schwefel]
	return functions[functionIndicator](offspring[0])


def mutation():
	global offspring
	global population

	offspring = population + np.random.standard_normal(N[NIndicator]) * sigma


def success():
	global offspring
	global population
	global successCount
	global sigma
	global lastFitness

	fitnessValue = fitness()
	if 0 <= fitnessValue < lastFitness:
		successCount += 1
		lastFitness = fitnessValue
		population = offspring

	if generationsCounter % 5 == 0:
		successRate = float(float(successCount) / float(5))
		if successRate > 1 / 5:
			sigma *= tau[tauIndicator]
		# print "success", lastFitness, "Round", generationsCounter, "Sigma", sigma
		else:
			sigma /= tau[tauIndicator]
		# print "failure", lastFitness, "Round", generationsCounter, "Sigma", sigma
		successCount = 0


worstFitnesses = [np.inf, np.inf, np.inf, np.inf]
functionIndicator = 0
functionNames = [sphere.__name__, rosenbrock.__name__, rastrigin.__name__, schwefel.__name__]


def main():
	global tauIndicator
	global NIndicator
	global functionIndicator
	global generationsCounter

	for functionIndicator in xrange(len(functionNames)):
		plots = [] * len(tau)
		plotsLegend = [] * len(tau)
		f, ax = plt.subplots(len(generations), len(N))
		f.suptitle(functionNames[functionIndicator])
		for NIndicator in xrange(len(N)):
			for generationsIndicator in xrange(len(generations)):
				for tauIndicator in xrange(len(tau)):
					init()
					fitnessLog = [0] * generations[generationsIndicator]
					for i in xrange(generations[generationsIndicator]):
						generationsCounter = i + 1
						mutation()
						success()  # does the selection, too, iff the fitness is better than the parents ones
						fitnessLog[i] = lastFitness
					print ("Function: " + functionNames[functionIndicator] + ", generations: " + str(
						generations[generationsIndicator]) + ", tau: " + str(tau[tauIndicator]) + ", Dimension: " + str(
						N[NIndicator]) + ", Mean: " + str(np.mean(fitnessLog)) + ", Std Deviation: " + str(
						np.std(fitnessLog)) + ",\n         Fitness: " + str(fitness()) + ", sigma: " + str(sigma))
					label = "Line" + str(tauIndicator)
					plot, = ax[generationsIndicator][NIndicator].plot(
						[np.inf if fitnesses == 0.0 else math.log(fitnesses, 10) for fitnesses in fitnessLog],
						label=label)
					plots.append(plot)
					plotsLegend.append("Tau: " + str(tau[tauIndicator]))
				ax[generationsIndicator][NIndicator].legend(plots, plotsLegend)
				ax[generationsIndicator][NIndicator].set_title("Dimension: " + str(N[NIndicator]))
				ax[generationsIndicator][NIndicator].set_ylabel("Fitness")
				ax[generationsIndicator][NIndicator].set_xlabel("Generations")
				plots = [] * len(tau)
				plotsLegend = [] * len(tau)
		f.subplots_adjust(hspace=0.5)
		manager = plt.get_current_fig_manager()
		manager.resize(*manager.window.maxsize())
	plt.show()


if __name__ == '__main__':
	main()
