# Genetic Algorithm
# Implement a genetic algorithm which uses the NumberAlloc class to represent individuals.
# The algorithm should evolve a population of individuals and return the best individual based on the highest fitness function.
# NumberAlloc is a class which represents a number allocation problem.
# 40 numbers are divided into 4 groups of 10 numbers each, with each group having a different value.
# The fitness function is the sum of the scores of the groups.

import random
import time
import typing
from copy import copy
from puzzle2 import puzzle2Piece, Tower
import numpy as np
from NumberAlloc import NumberAlloc

"""
P1:
Door, 6, 6, 5
Wall, 4, 9, 3
Lookout, 2, 3, 6

P2:
Door, 6, 6, 5
Wall, 4, 9, 3
Lookout, 2, 3, 6

C1:
Door, 6, 6, 5
Wall, 4, 9, 3
Wall, 4, 9, 3
Lookout, 2, 3, 6

C2:
Door, 6, 6, 5
Wall, 4, 9, 3
Wall, 4, 9, 3
Lookout, 2, 3, 6



NEw: 

Wall, 3, 8, 5
Lookout, 2, 5, 5
Wall, 1, 8, 9
Lookout, 6, 1, 2
Door, 5, 5, 2
Wall, 6, 4, 2

Lookout, 6, 1, 2
Wall, 2, 9, 6
Lookout, 2, 5, 5
Wall, 1, 8, 9


Children:
Wall, 3, 8, 5
, Lookout, 2, 5, 5
, Lookout, 6, 1, 2
, Lookout, 2, 5, 5
, Wall, 1, 8, 9

Lookout, 6, 1, 2
, Wall, 2, 9, 6
, Lookout, 6, 1, 2
, Lookout, 2, 5, 5
, Wall, 1, 8, 9


C1:
Wall, 3, 8, 5
Lookout, 2, 5, 5
Wall, 1, 8, 9
Door, 5, 5, 2
Wall, 1, 8, 9
Lookout, 6, 1, 2

dups: [3]



C2:
Lookout, 6, 1, 2
Wall, 2, 9, 6
Lookout, 2, 5, 5
Wall, 6, 4, 2



"""


def find_remove_indexes(data, original_frequencies: typing.Dict):
    new_freq = {}
    remove_indexes = []
    for i, piece in enumerate(data):
        if piece in new_freq:
            if new_freq[piece] >= original_frequencies[piece]:
                remove_indexes.append(i)
            else:
                new_freq[piece] += 1
        else:
            new_freq[piece] = 1

    return new_freq, remove_indexes


def fix_duplicates(numberlist, original_frequencies: typing.Dict[int, int]):
    new_freq, remove_indexes = find_remove_indexes(numberlist, original_frequencies)

    for num, count in original_frequencies.items():
        if num not in new_freq or count > new_freq[num]:
            numberlist[remove_indexes[0]] = num
            remove_indexes.pop(0)

    return numberlist

def fix_towers(p1_pieces, p2_pieces, original_frequencies: typing.Dict):
    # Check p1_pieces for duplicates based on original_frequencies and store indexes to remove
    new_freq, remove_indexes = find_remove_indexes(p1_pieces, original_frequencies)
    # Now that we have duplicates for P1, switch them
    len_diff = len(p1_pieces) - len(p2_pieces)
    for i in remove_indexes:
        _tmp = p1_pieces[i]
        p2_pieces[i-len_diff] = p1_pieces[i]
        p1_pieces[i] = _tmp

    # Check p2_pieces for duplicates based on original_frequencies and store indexes to remove
    new_freq, remove_indexes = find_remove_indexes(p2_pieces, original_frequencies)
    # Now that we have duplicates for P2, switch them
    for i in remove_indexes:
        _tmp = p2_pieces[i]
        p1_pieces[i+len_diff] = p2_pieces[i]
        p2_pieces[i] = _tmp

    return p1_pieces, p2_pieces


def check(piece, freq, original_frequencies):
    if piece in freq:
        if freq[piece] >= original_frequencies[piece]:
            return False
        else:
            return True
    elif piece in original_frequencies:
        return True
    return False


def uniform_random_xover(p1_list, p2_list, original_frequencies: typing.Dict):
    out1, out2 = [], []

    # For each position, choose randomly from the two parents, storing the unused piece in a list
    # If adding the random piece to the child would cause the frequency of the piece to exceed the original frequency,
    # then we can check if the piece from the other parent would also cause this issue. If not, we can add it to the child.
    # If the piece from the other parent would cause the frequency to exceed the original frequency,
    # then we can go down the unused list and keep trying to add pieces until we find one that doesn't cause the frequency to exceed the original frequency.
    # if none are found, we move on.

    # Loop twice to fill out1 and out2
    shorter = p1_list if len(p1_list) < len(p2_list) else p2_list
    for i in range(2):
        out = []
        unused = []
        frequencies = {}
        for i in range(len(shorter)):
            use_p1 = random.random() < 0.5
            piece, replacement = p1_list[i] if use_p1 else p2_list[i], p2_list[i] if use_p1 else p1_list[i]

            if check(piece, frequencies, original_frequencies):
                frequencies[piece] = frequencies.get(piece, 0) + 1
                out.append(piece)
                unused.insert(0, replacement)
                continue
            else:
                # Check the other piece
                if check(replacement, frequencies, original_frequencies):
                    frequencies[replacement] = frequencies.get(replacement, 0) + 1
                    out.append(replacement)
                    continue
                else:
                    for p in unused:
                        unused.remove(p)
                        if check(p, frequencies, original_frequencies):
                            frequencies[p] = frequencies.get(p, 0) + 1
                            out.append(p)
                            break

        if not out1:
            out1 = out
        else:
            out2 = out

    return out1, out2



def crossover(parent_1, parent_2, original_frequencies: typing.Dict):
    # To get two children, the number list from each parent is split in half.
    # The first half is used for the first child, and the second half is used for the second child.
    # The numbers in the first child are scanned for duplicates,
    # if any are found, the duplicate is swapped with the respective duplicate found in the second child.
    if isinstance(parent_1, NumberAlloc):
        p1_list = parent_1.numbers
        p2_list = parent_2.numbers
        parent1_first_half, parent1_second_half = p1_list[:-(len(p1_list) // -2)], p1_list[len(p1_list) // 2:]
        parent2_first_half, parent2_second_half = p2_list[:-(len(p2_list) // -2)], p2_list[len(p2_list) // 2:]

        c1_nums = parent1_first_half + parent2_second_half
        c2_nums = parent2_first_half + parent1_second_half

        c1_nums = fix_duplicates(c1_nums, original_frequencies)
        c2_nums = fix_duplicates(c2_nums, original_frequencies)

        return NumberAlloc(c1_nums), NumberAlloc(c2_nums)

    else:
        p1_list = parent_1.listOfPieces
        p2_list = parent_2.listOfPieces


        c1_nums, c2_nums = uniform_random_xover(p1_list, p2_list, original_frequencies)



        # c1_nums = p1_list[:-(len(p1_list) // -2)]
        #
        # for p in p2_list:
        #     if p not in c1_nums:
        #         c1_nums.append(p)
        #
        # c2_nums = p2_list[:-(len(p2_list) // -2)]
        #
        # for p in p1_list:
        #     if p not in c2_nums:
        #         c2_nums.append(p)


        # parent1_first_half, parent1_second_half = p1_list[:-(len(p1_list) // -2)], p1_list[-(len(p1_list) // -2):]
        # parent2_first_half, parent2_second_half = p2_list[:-(len(p2_list) // -2)], p2_list[-(len(p2_list) // -2):]
        #
        # c1_nums = parent1_first_half + parent2_second_half
        # c2_nums = parent2_first_half + parent1_second_half
        #
        # c1_nums, c2_nums = fix_towers(c1_nums, c2_nums, original_frequencies)
        #
        # if len(set(c1_nums)) != len(c1_nums) or len(set(c2_nums)) != len(c2_nums):
        #     print("Error: Duplicates in child")
        #     print(parent_1)
        #     print(parent_2)
        #     print(", ".join(str(x) for x in c1_nums))
        #     print(", ".join(str(x) for x in c2_nums))
        #     exit()

        return Tower(c1_nums, parent_1.allPieces), Tower(c2_nums, parent_1.allPieces)


def get_fitness_probabilities(population):
    # Calculate the fitness probabilities for each individual in the population
    # The fitness probabilities are calculated by dividing the fitness value by the sum of all fitness values
    # This ensures that the probabilities sum to 1
    # The fitness probabilities are then used to select the parents in the roulette wheel selection

    fitness_probabilities = []
    sum_of_fitness_values = sum(individual.fitness for individual in population)
    if sum_of_fitness_values == 0:
        return [1 / len(population) for individual in population]
    for individual in population:
        fitness_probabilities.append(individual.fitness / sum_of_fitness_values)

    return fitness_probabilities


def select_parents_roulette(sorted_population):
    # Select two parents from the sorted population using Roulette Wheel Selection
    # The two parents are selected using their fitness values

    # Choose two parents randomly weighted by their fitness values
    parent_1 = np.random.choice(sorted_population, p=get_fitness_probabilities(sorted_population))
    sorted_population.remove(parent_1)
    parent_2 = np.random.choice(sorted_population, p=get_fitness_probabilities(sorted_population))

    return parent_1, parent_2


def select_parents_tournament(sorted_population, k):
    # Select two parents from the sorted population
    # The two parents are selected using their fitness values
    # Because fitness values can be negative, we should use Tournament Selection
    # The k parameter is the number of individuals to be selected during the tournament, which are then compared to each other
    # The individual with the highest fitness is selected as the first parent
    # The tournament is re-run for the second parent

    choices = np.random.choice(sorted_population, replace=False, size=k)
    parent_1 = max(choices, key=lambda x: x.fitness)
    sorted_population.remove(parent_1)
    choices = np.random.choice(sorted_population, replace=False, size=k)
    parent_2 = max(choices, key=lambda x: x.fitness)

    return parent_1, parent_2


class GeneticAlgorithm:
    def __init__(self, data: typing.List[typing.Union[int, puzzle2Piece]], population_size: int, mutation_rate: float,
                 use_culling: bool, use_elitism: bool, max_time_seconds: float, tournament_size: int = 4):

        # Ensure there are exactly 40 numbers
        if len(data) != 40 and isinstance(data[0], int):
            raise ValueError("Data list must contain 40 numbers")
        elif len(data) > 1000 and isinstance(data[0], puzzle2Piece):
            raise ValueError("Data list must contain a max of 10 pieces")

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_time = max_time_seconds
        self.population = []
        self.fitness_history = []
        self.best_fitness = float('-inf')
        self.best_individual = None
        self.data = data
        self.convergence_threshold = 0.01
        self.data_frequency = {}
        self.tournament_size = tournament_size
        self.use_culling = use_culling
        self.use_elitism = use_elitism
        random.seed(1)
        np.random.seed(1)
        self.get_frequencies()

    def get_frequencies(self):
        for n in self.data:
            if n in self.data_frequency:
                self.data_frequency[n] += 1
            else:
                self.data_frequency[n] = 1

    def generate_population(self):
            for i in range(self.population_size):
                random.shuffle(self.data)

                if isinstance(self.data[0], int):
                    self.population.append(NumberAlloc(self.data))
                else:
                    self.population.append(Tower(self.data[:random.randint(2, len(self.data))], self.data))

    def run(self):
        # Initialize the population
        self.generate_population()

        # Loop through generations
        start_time = time.time()
        gen_num = 0
        next_gen = []

        print("Starting genetic algorithm...")

        best_fit = max(self.population, key=lambda x: x.fitness)
        self.best_fitness = best_fit.fitness
        self.best_individual = best_fit
        self.fitness_history.append(best_fit.fitness)

        # [Start] Generate random population of n chromosomes (suitable solutions for the problem)
        # [Fitness] Evaluate the fitness f(x) of each chromosome x in the population
        # Loop: [New population] Create a new population by repeating following steps until the new population is complete
        #
        # [Replace] Use new generated population for a further run of algorithm
        # [Test] If the end condition is satisfied, stop, and return the best solution in current population
        # [Loop] Go to step 2

        is_problem_1 = isinstance(self.best_individual, NumberAlloc)

        while time.time() - start_time < self.max_time:
            gen_num += 1

            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            if self.use_elitism:
                # Elistism
                # Sort the population by fitness
                # Select top 10% of the population to be added to next generation
                next_gen = sorted_population[:int(self.population_size * 0.1)]

            if self.use_culling:
                # Culling
                for i in sorted_population[int(self.population_size * 0.7):]:
                    self.population.remove(i)

            while len(next_gen) < self.population_size:

                # [Selection] Select two parent chromosomes from a population
                parent_1, parent_2 = select_parents_tournament(self.population, self.tournament_size) if is_problem_1 else select_parents_roulette(self.population)

                # [Crossover] With a crossover probability cross over the parents to form a new offspring (children).
                # If no crossover was performed, offspring is an exact copy of parents.
                child_1, child_2 = crossover(parent_1, parent_2, self.data_frequency)

                # if max(child_1.fitness, child_2.fitness) > max(parent_1.fitness, parent_2.fitness):
                #     print("Child fitness is higher than parent fitness..")
                #     print(parent_1)
                #     print(parent_2)
                #     print(child_1)
                #     print(child_2)

                # [Mutation] With a mutation probability mutate new offspring at each locus (position in chromosome).
                child_1.mutate(self.mutation_rate)
                child_2.mutate(self.mutation_rate)


                # [Accepting] Place new offspring and parents in a new population
                next_gen.append(child_1)
                next_gen.append(child_2)
                # next_gen.append(parent_1)
                # next_gen.append(parent_2)

                # Remove parents from population
                # self.population.remove(parent_1)
                # self.population.remove(parent_2)


            best_fit = max(next_gen, key=lambda x: x.fitness)
            self.fitness_history.append(best_fit.fitness)

            # [Test] If the end condition is satisfied, stop, and return the best solution in current population
            if best_fit.fitness > self.best_fitness:
                self.best_fitness = best_fit.fitness
                self.best_individual = best_fit
                print(f"Generation {gen_num} | Best Fitness: {self.best_fitness:,} | Worst Fitness: {min(self.fitness_history)}")

            else:
                print(f"Generation {gen_num} | Best Fitness: {self.best_fitness}")
                # # Check if fitness has converged
                # if len(self.fitness_history) > 1:
                #     if abs(self.fitness_history[-1] - self.fitness_history[-2]) < self.convergence_threshold:
                #         print(f"Converged after {gen_num} generations")
                #         return self.best_individual

            # [Replace] Replace the old population with the new population
            self.population = next_gen
            next_gen = []

        print(f"Max time reached after {gen_num} generations")
        return self.best_individual









        #     # Elitism
        #     # Find all individuals in the top 10% of the population
        #     # and add them to the next generation
        #     # This is done by sorting the population by fitness
        #     # and then selecting the top 10%
        #     _pop = copy(self.population)
        #     _pop.sort(key=lambda x: x.fitness, reverse=False)
        #     # next_gen = _pop[:int(self.population_size * 0.1)]
        #
        #     # Culling
        #     # Remove the bottom 10% of the population
        #     _culled_pop = _pop[int(len(_pop) * 0.1):]
        #
        #     # Print the fitness of each individual in the population
        #     print("Fitness: ", [x.fitness for x in _culled_pop])
        #
        #     # Select two parents from the culled population
        #     # Each individual is assigned a chance to be selected based on its fitness
        #     # The chance of being selected is proportional to its fitness
        #     # where fitness values are normalized to a range of 0 to 1
        #     # The higher the fitness, the higher the chance of being selected
        #
        #     # Save max and min fitness values
        #     max_fitness = _culled_pop[0].fitness
        #     min_fitness = _culled_pop[-1].fitness
        #     min_p, max_p = 0.1, 0.9
        #     _pop_fitness_normalized = [((max_p-min_p)*((x.fitness - min_fitness) / (max_fitness - min_fitness)))+min_p for x in _culled_pop]
        #     print(f"Fitness normalized (between {min_p}-{max_p}):\n{_pop_fitness_normalized}")
        #
        #     # if summed_fitness <= 0:
        #     #     # Choose two random individuals, where indviduals are weighted by their fitness
        #     #     # Fitness values can be negative, so we need to make sure we don't divide by zero
        #     #     parent_1 = random.choices(_pop, weights=[x.fitness for x in _pop])[0]
        #     #
        #     #     rand_num = random.uniform(0, summed_fitness)
        #     #     for i in range(len(_pop)):
        #     #         rand_num -= _pop[i].fitness
        #     #         if rand_num <= 0:
        #     #             break
        #     #     _parents = random.choices(_pop, weights=[abs(min(x.fitness)])
        #     # else:
        #     _parents = np.random.choice(_culled_pop, size=2, replace=False, p=_pop_fitness_normalized)
        #     parent_1, parent_2 = _parents[0], _parents[1]
        #
        #     # Crossover
        #     # Create two new individuals by combining the genes of the parents
        #     # The new individuals are distinct children of the parents
        #     child_1, child_2 = crossover(parent_1, parent_2)
        #
        #     # Mutation
        #     # Each gene in the new individuals is given a chance to mutate
        #     # The chance of mutation is determined by the mutation rate
        #     # If the gene mutates, a random number is chosen from the numbers list
        #     # and the gene is replaced with the number
        #     child_1.mutate(self.mutation_rate)
        #     child_2.mutate(self.mutation_rate)
        #
        #     # Add the new individuals to the next generation
        #     next_gen = [child_1, child_2, parent_1, parent_2]
        #
        #     # Store the best fitness of this generation in the history
        #     self.fitness_history.append(self.population[0].fitness)
        #
        #     # Replace the current population with the next generation
        #     self.population = next_gen
        #     print(f"Generation {len(self.fitness_history)} | Population size: {len(self.population)}")
        #
        #     # Store the best individual of this generation
        #     if self.population[0].fitness > self.best_fitness:
        #         self.best_fitness = self.population[0].fitness
        #         self.best_individual = self.population[0]
        #
        # # Print the fitness history for each generation
        # for i in range(len(self.fitness_history)):
        #     print(f"Generation {i}: {self.fitness_history[i]}")
        #
        # # Print the best individual of the final generation
        # print(f"Best individual: {self.best_individual}")
        # print(f"Best fitness: {self.best_individual.fitness}")
        # print(f"Time taken: {time.time() - start_time}")
        #
        # # Return the best individual of the last generation
        # return self.best_individual





            














# def genetic(population, fitness, mutation_rate, crossover_rate, max_generations):
#     # Initialize the population
#     population = initialize_population(population, fitness)
#     # Loop through generations
#     for i in range(max_generations):
#         # Select the parents
#         parents = select_parents(population, fitness)
#         # Crossover
#         children = crossover(parents, crossover_rate)
#         # Mutate
#         children = mutate(children, mutation_rate)
#         # Replace the population
#         population = children
#     # Return the best solution
#     return population[0]
#
#
# def initialize_population(population, fitness):
#     # Initialize the population
#     population = []
#     for i in range(0, len(fitness)):
#         population.append(fitness[i][0])
#     return population
#
#
# def select_parents(population, fitness):
#     # Select the parents
#     parents = []
#     for i in range(0, len(fitness)):
#         parents.append(population[fitness[i][1]])
#     return parents
#
#
# def crossover(parents, crossover_rate):
#     # Crossover
#     children = []
#     for i in range(0, len(parents)):
#         if random.random() < crossover_rate:
#             children.append(crossover_two(parents[i], parents[(i + 1) % len(parents)]))
#         else:
#             children.append(parents[i])
#     return children
#
#
# def crossover_two(parent1, parent2):
#     # Crossover two parents
#     child = []
#     for i in range(0, len(parent1)):
#         if random.random() < 0.5:
#             child.append(parent1[i])
#         else:
#             child.append(parent2[i])
#     return child
#
#
# def mutate(children, mutation_rate):
#     # Mutate
#     for i in range(0, len(children)):
#         for j in range(0, len(children[i])):
#             children[i][j] = children[i][j].get_mutated_copy(mutation_rate)
#     return children
