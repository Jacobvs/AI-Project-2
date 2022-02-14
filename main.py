import random
import sys

import numpy as np

from genetic import GeneticAlgorithm
from NumberAlloc import NumberAlloc
from puzzle2 import puzzle2Piece


def create_test_file(filename, is_problem_1: bool):
    # Create a file with the given filename
    # The file contains 40 random integers between -10 and 10, each on a separate line
    # The file is created in the current directory
    # The file is not returned
    with open(filename, 'w') as f:
        if is_problem_1:
            for i in range(40):
                f.write(str(random.randint(-10, 10)) + '\n')
        else:
            for i in range(random.randint(3, 10)):
                type = np.random.choice(['Door', 'Wall', 'Lookout'])
                width = random.randint(1, 9)
                strength = random.randint(1, 9)
                cost = random.randint(1, 9)
                f.write(f"{type}\t{width}\t{strength}\t{cost}\n")


# create_test_file('problem2.txt', False)

def get_numbers(filename):
    # Read an input file with 40 numbers on each line
    # The numbers are returned as a list of integers
    # If the file does not exist, an error is raised
    with open(filename, 'r') as f:
        numbers = []
        for line in f:
            numbers.append(int(line))
    return numbers

def get_pieces(filename):
    with open(filename, 'r') as f:
        pieces = []
        for line in f:
            attrs = line.split('\t')
            piece = puzzle2Piece(attrs[0], int(attrs[1]), int(attrs[2]), int(attrs[3]))
            print(piece)
            pieces.append(piece)
        return pieces


def main(*args):
    if len(args) == 3:
        problem = int(args[0])
        input_file = args[1]
        max_runtime_seconds = int(args[2])
        if problem == 1:
            nums = get_numbers(input_file)
            GA = GeneticAlgorithm(data=nums, population_size=20, mutation_rate=0.15, tournament_size=4,
                                  use_elitism=False, use_culling=False, max_time_seconds=max_runtime_seconds)
            gen_num, best = GA.run()
            bins = '\n'.join(str(k) + ": " + str(v) for k, v in best.bins.items())
            print(f"Best Individual:\nScore: {best.fitness:,}\nBins:\n{bins}")
            print(f"Max Generation Reached: {gen_num}")
        elif problem == 2:
            pieces = get_pieces(input_file)
            GA2 = GeneticAlgorithm(data=pieces, population_size=600, mutation_rate=0.3,
                                   use_elitism=False, use_culling=False, max_time_seconds=10)
            gen_num, best2 = GA2.run()

            # Print Best Individual
            print(f"Best Individual:\nScore: {best2.fitness:,}\nStack:\n{best2}")
            print(f"Max Generation Reached: {gen_num}")

    elif len(args) == 0:
        run_p1 = input("Run Problem 1 or Problem 2? (1/2)")
        p1 = True
        if run_p1 == '2':
            p1 = False

        gen_new = input("Generate new test file? (y/n)")
        if gen_new == 'y':
            if p1:
                create_test_file('problem1.txt', True)
            else:
                create_test_file('problem2.txt', False)

        if p1:
            nums = get_numbers('problem1.txt')

            GA = GeneticAlgorithm(data=nums, population_size=20, mutation_rate=0.15, tournament_size=4,
                                  use_elitism=False, use_culling=False, max_time_seconds=10)
            gen_num, best = GA.run()

            # Print Best Individual
            print(f"Best Individual:\n\tScore: {best.fitness:,}\n\tBins: {best.bins}")

        else:
            pieces = get_pieces('problem2.txt')
            GA2 = GeneticAlgorithm(data=pieces, population_size=500, mutation_rate=0.3,
                                   use_elitism=False, use_culling=False, max_time_seconds=10)
            best2 = GA2.run()

            # Print Best Individual
            print(f"Best Individual:\n\tScore: {best2.fitness:,}\n\tStack: {best2}")
    else:
        print("Usage: main.py <problem (1/2)> <input_file> <max_runtime_seconds>")
        return

if __name__ == "__main__":
    main(*sys.argv[1:])
