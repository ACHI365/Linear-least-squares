import random

import numpy as np
from matplotlib import pyplot as plt

from QRFactorization import solve, reduced_factorization, modified_Gram_Schmidt, carsToMatrix, houseHolder

ageConstant = 2.2
techConstant = 1.2


class Car:
    def __init__(self, owner, name, age, techCheck):
        self.condition = 100
        self.age = age
        self.name = name
        self.techCheck = techCheck
        self.owner = owner
        self.calculateCondition()

    def getName(self):
        return self.name

    def getOwner(self):
        return self.owner

    def getTechCheck(self):
        return self.techCheck

    def getAge(self):
        return self.age

    def calculateCondition(self):  # I made up this equation with most accurate results
        self.condition -= int(ageConstant * self.age)
        self.condition += int(techConstant * self.techCheck)

        if self.condition > 100:
            self.condition = 100

    def getCondition(self):
        return self.condition

    def __str__(self):
        return f"{self.owner} | Name: {self.name} | Age: {self.age} | Tech Check per year: {self.techCheck} | " \
               f"Condition: {self.condition} "


def generateCars(number, firstName, lastName, carName):
    # get names from data
    with open(firstName) as f:
        lines = f.readlines()
        firstNames = list(map(lambda name: name[:len(name) - 1], lines))

    with open(lastName) as f:
        lines = f.readlines()
        lastNames = list(map(lambda surname: surname[:len(surname) - 1], lines))

    with open(carName) as f:
        lines = f.readlines()
        carNames = list(map(lambda cars: cars[:len(cars) - 1], lines))

    i = 0
    carsList = []

    while i < number:
        fName = firstNames[random.randint(0, len(firstNames) - 1)]

        fullName = fName + " " + lastNames[random.randint(0, len(lastName) - 1)]

        carName = carNames[random.randint(0, len(carNames) - 1)]

        age = random.randint(0, 40)
        techCheck = random.randint(0, 6)

        carsList.append(Car(fullName, carName, age, techCheck))
        i += 1

    return carsList


print("Choose The way to solve problem:")
print("Modified Gram-Schmidt - 0")
print("Householder - 1")
print("exit - 2")

option = int(input("Choose: "))

carsGen = generateCars(100, "Data\\firstNames.txt", "Data\\lastNames.txt", "Data\\cars.txt")

# print(*cars, sep='\n')
A, b = carsToMatrix(carsGen)
_x = []

# Cx = d
C = np.array([7, 8])
d = np.array([11])

while True:
    if option == 0:
        # combining arrays
        D = np.row_stack([A, C])
        #
        Q, R = modified_Gram_Schmidt(D)
        # separating Q1 for A and Q2 for C
        Q1 = Q[:D.shape[0] - 1]
        Q2 = Q[D.shape[0] - 1:]

        _Q, _R = modified_Gram_Schmidt(Q2.T)

        _x = solve(R, Q1, Q2, _Q, _R, d, b)
    elif option == 1:
        D = np.row_stack([A, C])
        Q, R = houseHolder(D)
        Q, R = reduced_factorization(Q, R, D)

        Q1 = Q[:D.shape[0] - 1]
        Q2 = Q[D.shape[0] - 1:]

        _Q, _R = houseHolder(Q2.T)
        _Q, _R = reduced_factorization(_Q, _R, Q2.T)

        _x = solve(R, Q1, Q2, _Q, _R, d, b)
    elif option == 2:
        break
    else:
        raise ValueError("Wrong input")

    carCustom = Car("Car Owner", "GMC", 26, 4)
    carVector = np.array([carCustom.getAge(), carCustom.getTechCheck()])
    result = int(np.matmul(_x, carVector))

    print(carCustom.getName(), "of", carCustom.getOwner(), "will be in", result, "condition")

    print("Try again with different route, or type 2 for exit")

    option = int(input("Choose: "))

"""
    Unfortunately, generated data seems a little of but it is actually correct. the problem is that matplotlib library
    cannot visualize 3+ dimensional graphs, since it has only 2D visuals
"""

trainedX = list(map(lambda car: car[0], enumerate(carsGen)))
trainedX = np.array(trainedX)

trainedY = list(map(lambda car: car.getCondition(), carsGen))
trainedY = np.array(trainedY)

newData = generateCars(50, "Data\\firstNames.txt", "Data\\lastNames.txt", "Data\\cars.txt")

predictX = list(map(lambda car: car[0], enumerate(newData)))
predictX = np.array(predictX)

print(*newData, sep="\n")

predictY = list(map(lambda car: int(np.matmul(_x, np.array([car.getAge(), car.getTechCheck()]))), newData))
predictY = np.array(predictY)

plt.plot(trainedX, trainedY, 'o')
plt.plot(predictX, predictY, 'o', color='r')
plt.show()
