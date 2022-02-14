import enum

from genetic import *

class pieceType(enum.Enum):
    Door = "Door"
    Wall = "Wall"
    Lookout = "Lookout"

class puzzle2Piece:
    """
    Class for puzzle 2 pieces
    """
    def __init__(self, pieceType, width, strength, cost):
        self.pieceType = pieceType
        self.width = width
        self.strength = strength
        self.cost = cost

class tower:
    """
    Class for tower
    Consists of n tower pieces stored in a list
    Last index in the list being the last piece
    """
    def __init__(self, listOfPieces, isValid):
        self.listOfPieces = listOfPieces
        self.isValid = isValid;


def findTowerCost(towerUsed) -> float:
    totalCost = 0
    for puzzle2Piece in towerUsed:
        totalCost + puzzle2Piece.cost
    return totalCost


def isTowerValid(listOfUsedPieces):
    """
    Determines if the tower is valid, considering project specifications
    """
    lastPiece = listOfUsedPieces[0]  # initialize first piece for width check
    # loop throughout tower pieces to make sure there isn't a larger piece on top of a smaller one
    # and each piece can hold the weight above it

    listSize = len(listOfUsedPieces)

    for i, puzzle2Piece in enumerate(listOfUsedPieces):
        #  make sure this piece is not wider than the previous
        if puzzle2Piece.width > lastPiece.width:
            return False
        lastPiece = puzzle2Piece;

        #  make this piece can hold the rest of the building
        if (listSize-(i+1)) > puzzle2Piece.strength:
            return False

    #  make sure the top and bottom are correct and all pieces inbetween are walls
    if not listOfUsedPieces[-1].pieceType == "Lookout" or not listOfUsedPieces[0].pieceType == "Door":
        return False

    if any(x.pieceType != "Wall" for x in listOfUsedPieces[1:-1]): return False

    return True


def calcTowerScore(towerUsed) -> float:
    """returns the tower's fitness"""
    if not towerUsed.isValid:
        return 0
    return 10 + (len(towerUsed.listOfPieces))**2 + findTowerCost(towerUsed)
