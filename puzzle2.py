import enum

from genetic import *

# class pieceType(enum.Enum):
#     Door = "Door"
#     Wall = "Wall"
#     Lookout = "Lookout"


class puzzle2Piece:
    """
    Class for puzzle 2 pieces
    """
    def __init__(self, pieceType, width, strength, cost):
        self.pieceType = pieceType
        self.width = width
        self.strength = strength
        self.cost = cost

    def __eq__(self, other):
        if self.pieceType == other.pieceType and self.width == other.width and \
            self.strength == other.strength and self.cost == other.cost:
            return True
        return False

    def __hash__(self):
        return hash((self.pieceType, self.width, self.strength, self.cost))

    def __str__(self):
        return f"{self.pieceType}, {self.width}, {self.strength}, {self.cost}\n"


class Tower:
    """
    Class for tower
    Consists of n tower pieces stored in a list
    Last index in the list being the last piece
    """

    def __init__(self, listOfPieces, allPieces):
        self.listOfPieces = listOfPieces
        self.allPieces = allPieces
        self.isValid = isTowerValid(listOfPieces)
        self.fitness = self.calcTowerScore()

    def __str__(self):
        return ''.join(str(piece) for piece in self.listOfPieces)

    def calcTowerScore(self) -> float:
        """returns the tower's fitness"""
        if self.isValid is False:
            return 0
        return max(0, 10 + (len(self.listOfPieces) ** 2) - self.findTowerCost())

    def findTowerCost(self) -> float:
        totalCost = 0
        for puzzle2Piece in self.listOfPieces:
            totalCost += puzzle2Piece.cost
        return totalCost

    # def calcTowerFitness(self) -> float:
    #     " add +10 to the score for each piece not on the end which is a wall"
    #     fitness = self.findTowerCost()
    #     nonwall = len([x.pieceType != 'Wall' for x in self.listOfPieces])
    #     fitness -= nonwall
    #     fitness += len(self.listOfPieces) if self.listOfPieces[-1].pieceType == "Lookout" else 0
    #     fitness += len(self.listOfPieces) if self.listOfPieces[0].pieceType == "Door" else 0
    #     return fitness

    def mutate(self, mutation_rate):
        """mutates the tower"""
        if random.random() < mutation_rate:
            # Either Add or Remove a piece
            remaining = set(self.allPieces) - set(self.listOfPieces)
            if (len(remaining) == 0 or random.random() < 0.5) and len(self.listOfPieces) > 1:
                # Remove a random piece
                self.listOfPieces.pop(random.randint(0, len(self.listOfPieces) - 1))
            else:
                # Add a random piece
                self.listOfPieces.append(random.choice(list(remaining)))

        for i, piece in enumerate(self.listOfPieces):
            if random.random() < mutation_rate:
                # move the piece somewhere else
                new_index = random.randint(0, len(self.listOfPieces) - 1)
                self.listOfPieces[i], self.listOfPieces[new_index] = self.listOfPieces[new_index], self.listOfPieces[i]

        self.isValid = isTowerValid(self.listOfPieces)
        self.fitness = self.calcTowerScore()




def isTowerValid(listOfUsedPieces: typing.List[puzzle2Piece]) -> bool:
    """
    Determines if the tower is valid, considering project specifications
    """
    # loop throughout tower pieces to make sure there isn't a larger piece on top of a smaller one
    # and each piece can hold the weight above it
    lastPiece = None
    listSize = len(listOfUsedPieces)

    for i, piece in enumerate(listOfUsedPieces, start=1):
        if i != 1:
            #  make sure this piece is not wider than the previous
            if piece.width > lastPiece.width:
                return False

            #  make this piece can hold the rest of the building
            if (listSize-i) > piece.strength:
                return False
        lastPiece = piece

    #  make sure the top and bottom are correct and all pieces inbetween are walls
    if not listOfUsedPieces[-1].pieceType == "Lookout" or not listOfUsedPieces[0].pieceType == "Door":
        return False

    if any(x.pieceType != "Wall" for x in listOfUsedPieces[1:-1]):
        return False

    return True



