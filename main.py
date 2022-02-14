import random

from genetic import GeneticAlgorithm
from NumberAlloc import NumberAlloc


def create_test_file(filename):
    # Create a file with the given filename
    # The file contains 40 random integers between -10 and 10, each on a separate line
    # The file is created in the current directory
    # The file is not returned
    with open(filename, 'w') as f:
        for i in range(40):
            f.write(str(random.randint(-10, 10)) + '\n')



# create_test_file('test.txt')



def get_numbers(filename):
    # Read an input file with 40 numbers on each line
    # The numbers are returned as a list of integers
    # If the file does not exist, an error is raised
    with open(filename, 'r') as f:
        numbers = []
        for line in f:
            numbers.append(int(line))
    return numbers

nums = get_numbers('test.txt')
# best = NumberAlloc(nums)
# print(f"Best Individual:\n\tFitness: {best.fitness}\n\tBins: {best.bins}")
# best.mutate(0.5)
# print(f"Best Individual:\n\tFitness: {best.fitness}\n\tBins: {best.bins}")

GA = GeneticAlgorithm(nums, 20, 0.2, 10)
best = GA.run()

# Print Best Individual
print(f"Best Individual:\n\tFitness: {best.fitness}\n\tBins: {best.bins}")
