import enum
import math
import typing
from asyncio import sleep
from copy import copy
import random
import Node


class BinType(enum.Enum):
    """
    Enum class for BinType
    """
    MULTIPLY = 1
    ADD = 2
    SUBTRACT = 3
    IGNORED = 4

    def get_score(self, numbers) -> typing.Union[int, float]:
        """
        :param numbers:
        :return: int of the scored bin
        """
        # Multiply all numbers together
        if self == BinType.MULTIPLY:
            return math.prod(numbers)
        # Add numbers together
        elif self == BinType.ADD:
            return sum(numbers)
        # Subtract the smallest number from the largest in numbers
        elif self == BinType.SUBTRACT:
            return max(numbers) - min(numbers)
        # Ignore the numbers
        elif self == BinType.IGNORED:
            return 0


class NumberAlloc:
    """
    Class for number allocation
    Scores bins of numbers based on BinType
    """

    def __init__(self, numbers):
        """
        :param numbers: A list of 40 numbers to be used in the genetic algorithm
        """
        self.numbers = numbers
        self.bins: typing.Dict[BinType: typing.List[int]] = {
            BinType.MULTIPLY: numbers[:10],
            BinType.ADD: numbers[10:20],
            BinType.SUBTRACT: numbers[20:30],
            BinType.IGNORED:  numbers[30:]
        }
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        """
        :return: A summed score from all bins to get a fitness score
        """
        score = 0
        for bin_type in self.bins.keys():
            score += bin_type.get_score(self.bins[bin_type])
        return score

    def mutate(self, mutation_rate):
        """
        Iterate through each number in the bins
         - use the mutation rate to determine if it should be swapped randomly with a number in another bin
        :param mutation_rate: mutation chance in the range of 0.0 to 1.0
        :return: A mutated copy of the NumberAlloc object
        """
        for bin_type in self.bins.keys():
            for i in range(10):
                # If a random value is less than the mutation rate, swap the number with a random number in another bin
                if random.random() < mutation_rate:
                    # Choose a random bin except the current bin
                    random_bin_type = random.choice(list(filter(lambda x: x != bin_type, self.bins.keys())))
                    # Choose a random number from the random bin
                    random_num = random.randint(0, len(self.bins[random_bin_type]) - 1)
                    # Switch the current number with a random number from the random bin
                    _tmp = self.bins[bin_type][i]
                    self.bins[bin_type][i] = self.bins[random_bin_type][random_num]
                    self.bins[random_bin_type][random_num] = _tmp

        self.fitness = self.calculate_fitness()

