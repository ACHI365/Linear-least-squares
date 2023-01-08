import random

import numpy as np
from matplotlib import pyplot as plt

from QRFactorization import least_square_GS, least_square_house, playersToMatrix
from normalEquations import normal_equation

CONSTANT = 1.2


class Player:
    def __init__(self, age, fullName, gender, workHours, rating):
        self.potential = None
        if gender.lower() == "f":
            self.gender = "Female"
        else:
            self.gender = "Male"
        self.age = age
        self.fullName = fullName
        self.workHours = workHours
        self.rating = rating
        self.calculatePotential()

    def getGender(self):
        return self.gender

    def getFullName(self):
        return self.fullName

    def getWorkHours(self):
        return self.workHours

    def getAge(self):
        return self.age

    def getRating(self):
        return self.rating

    def calculatePotential(self):  # I made up this equation with most accurate results
        self.potential = self.rating + int(
            0.65 * CONSTANT * self.workHours + ((23 - self.age) * ((71 - self.rating) / 20)))
        if self.potential > 96:
            self.potential = 96

    def getPotential(self):
        return self.potential

    def __str__(self):
        return f"{self.fullName} | Age: {self.age} | Gender: {self.gender} | Work Hours: {self.workHours} | Rating:" \
               f" {self.rating} | Potential: {self.potential}"


def generatePlayers(number, firstName, lastName):
    # get names from data
    with open(firstName) as f:
        lines = f.readlines()
        firstNames = list(map(lambda name: name[:len(name) - 1], lines))

    with open(lastName) as f:
        lines = f.readlines()
        lastNames = list(map(lambda surname: surname[:len(surname) - 1], lines))

    i = 0
    playerList = []
    # create n players
    while i < number:
        fName = firstNames[random.randint(0, len(firstNames) - 1)]
        gender = fName[2]

        fullName = fName + " " + lastNames[random.randint(0, len(lastName) - 1)]

        age = random.randint(16, 23)
        workHrs = random.randint(8, 30)
        rating = random.randint(40, 70)

        playerList.append(Player(age, fullName, gender, workHrs, rating))
        i += 1

    return playerList


print("Choose The way to solve problem:")
print("Normal Equation - 0")
print("Modified Gram-Schmidt - 1")
print("Householder - 2")
print("exit - 3")

option = int(input("Choose: "))

players = generatePlayers(100, "Data\\firstNames.txt", "Data\\lastNames.txt")

A, b = playersToMatrix(players)
_x = []

while True:
    if option == 0:
        _x = normal_equation(A, b)
    elif option == 1:
        _x = least_square_GS(A, b)
    elif option == 2:
        _x = least_square_house(A, b)
    elif option == 3:
        break
    else:
        raise ValueError("Wrong input")

    playerCustom = Player(21, "John Doe", "M", 27, 70)
    playerVector = np.array([playerCustom.getAge(), playerCustom.getRating(), playerCustom.getWorkHours()])
    result = int(np.matmul(_x, playerVector))

    print(playerCustom.getFullName(), "will get to", result, "OVR")

    print("Try again with different route, or type 3 for exit")

    option = int(input("Choose: "))

"""
    Unfortunately, generated data seems a little of but it is actually correct. the problem is that matplotlib library
    cannot visualize 3+ dimensional graphs, since it has only 2D visuals
"""

trainedX = list(map(lambda player: player[0], enumerate(players)))
trainedX = np.array(trainedX)

trainedY = list(map(lambda player: player.getPotential(), players))
trainedY = np.array(trainedY)

newData = generatePlayers(50, "Data\\firstNames.txt", "Data\\lastNames.txt")

predictX = list(map(lambda player: player[0], enumerate(newData)))
predictX = np.array(predictX)

print(*newData, sep="\n")

predictY = list(
    map(lambda player: int(np.matmul(_x, np.array([player.getAge(), player.getRating(), player.getWorkHours()]))),
        newData))
predictY = np.array(predictY)

plt.plot(trainedX, trainedY, 'o')
plt.plot(predictX, predictY, 'o', color='r')
plt.show()
