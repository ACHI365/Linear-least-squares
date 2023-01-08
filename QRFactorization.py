import numpy as np
from normalEquations import euclidean_length


# convert vector to unit vector
def convert_to_unit(vector):
    return vector / euclidean_length(vector)


# round numbers which are really low to 0
def roundMatrix(A):
    tol = 1e-10
    A.real[abs(A.real) <= tol] = 0.0
    return A


# takes row vector as an input, turns it into column vector and transforms back to row
# both return variables are row vectors
def modified_Gram_Schmidt(A):
    A = A.T
    r = np.zeros((len(A), len(A)))
    q = np.zeros((len(A), len(A[0])))

    for j in range(len(A)):
        y = A[j]
        for i in range(j):
            r[i][j] = np.dot(np.transpose(convert_to_unit(q[i])), y)
            y = y - np.dot(r[i][j], q[i])
        r[j][j] = euclidean_length(y)

        if r[j][j] != 0:
            q[j] = y / r[j][j]
        else:
            q[j] = y

    return q.T, r


# full_QR factorization with modified_gram_schmidt algorithm
# (turns out LS problem could be solved without this, but I still implemented) (output is same as Householder)
# takes in matrix with row vectors as an argument and returns row vectors in return
def full_QR(A):
    # change to column vector
    A = A.T
    shape = (A.shape[1], A.shape[1])
    # creates square shaped matrix which is needed for full factorization
    res = np.zeros(shape)
    # fills already known entries
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res[i][j] = A[i][j]

    counter = 0
    # add vectors that are linearly independent by checking ranks
    for i in range(res.shape[0] - A.shape[0]):
        i = A.shape[0] + i
        rank = np.linalg.matrix_rank(res)
        while True:
            for j in range(res.shape[1]):
                if j == counter:
                    res[i][j] = 1
                else:
                    res[i][j] = 0
            if np.linalg.matrix_rank(res) > rank:
                break
            else:
                counter += 1
    # after filling the matrix make QR factorization with modified Gram Schmidt
    Q, R = modified_Gram_Schmidt(res.T)

    # remove unnecessary entries
    R = (R.T[:A.shape[0]]).T

    Q = roundMatrix(Q)
    return Q, R


# Rx = QT * b must be solved
# takes in matrix with row vectors
def least_square_QR(A, b, Q, R):
    R = R[:A.shape[1]]
    # get QR with full factorization remove everything except upper sub-matrix from R
    RI = np.linalg.inv(R)  # to solve x, we should rewrite as x = R^-1 * QT *b
    QTb = np.dot(np.transpose(Q), b)
    QTb = QTb[:R.shape[0]]
    # calculate Qtb and then remove everything below length of vector

    return np.dot(RI, QTb)


def least_square_reduced(A, b):
    Q, R = modified_Gram_Schmidt(A)
    return least_square_QR(A, b, Q, R)


def least_square_GS(A, b):
    Q, R = full_QR(A)
    return least_square_QR(A, b, Q, R)


def least_square_house(A, b):
    Q, R = houseHolder(A)
    return least_square_QR(A, b, Q, R)


# takes row matrix and removes row/columns
def reduce_matrix(A):
    A = A[1:]
    res = np.zeros((A.shape[0], A.shape[1] - 1))
    for j in range(A.shape[0]):
        res[j] = A[j][1:]

    return roundMatrix(res)


# takes row matrix and fills it with ii row/columns of identity matrix
def fulfill_matrix(H, i):
    if i == 0:
        return H
    else:
        I = np.identity(len(H) + i)
        for j in range(i, len(I)):
            for k in range(i, len(I[0])):
                I[j][k] = H[j - i][k - i]

        return I


# H = I - 2vvT/vTv
# takes in row matrix
def houseHolder(A):
    AT = A.T
    Hs = []
    i = 0
    while i < A.shape[1]:
        # take first column vector
        x = AT[0]

        # find its similar vector based on length
        w = np.array([0] * x.shape[0])

        w[0] = 1
        w = euclidean_length(x) * w

        v = w - x

        I = np.identity(len(x))
        vT = np.transpose(v)

        # find reflection over the subtraction of above-mentioned vectors
        P = np.outer(v, vT) / np.dot(vT, v)
        H = I - 2 * P

        Hs.append(fulfill_matrix(H, i))

        # calculate matrix for new A
        newA = H @ AT.T

        # used to round really low values to 0 to check for upper triangular matrix
        newA = roundMatrix(newA)

        newAT = reduce_matrix(newA.T)

        if newAT.shape[0] == 1 and newAT.shape[1] == 1:
            break

        AT = newAT

        i += 1

    # R is last matrix and Q will be H1*H2...*Hn
    if len(Hs) == 0:
        Hs.append(np.identity(A.shape[1]))
        R = Hs[0] @ A
    else:
        R = A
        for H in Hs:
            R = H @ R
        R = roundMatrix(R)

    Q = Hs[0]
    for p in range(1, len(Hs)):
        Q = Q @ Hs[p]
    Q = roundMatrix(Q)

    return Q, R


"""
1.  Compute two QR factorizations
    [A C] = [Q1 Q2] R ---> Q2.T = _Q @ _R
    example:
    A = np.row_stack([A, C])
    Q, R = modified_Gram_Schmidt(A)
    # separating Q1 for A and Q2 for C
    Q1 = Q[:A.shape[0] - 1]
    Q2 = Q[A.shape[0] - 1:]
2.  solve _R.T @ u = d and then c = _Q.T @ Q1.T @ b - u
3.  solve _R.T @ w = c and then y = Q1.T @ b - Q2.T @ w
4.  compute R @ _x = y   
"""


def reduced_factorization(Q, R, A):
    for i in range(A.shape[0] - A.shape[1]):
        m = Q.shape[1]
        Q = np.delete(Q, m - 1, 1)

    for i in range(A.shape[0] - A.shape[1]):
        m = R.shape[0]
        R = np.delete(R, m - 1, 0)

    return Q, R


def solve(R, Q1, Q2, _Q, _R, d, b):
    u = np.linalg.solve(_R.T, d)

    c = _Q.T @ Q1.T @ b - u

    w = np.linalg.solve(_R, c)

    y = Q1.T @ b - Q2.T @ w

    _x = np.linalg.solve(R, y)

    return _x


# turns player list into matrix (for playerRatings.py)
def playersToMatrix(Players):
    A = []
    b = []
    for player in Players:
        row = [player.getAge(), player.getRating(), player.getWorkHours()]
        A.append(row)
        b.append(player.getPotential())
    b = np.array(b)
    A = np.array(A)
    return A, b


# turns cars list into matrix (for carCondition.py)
def carsToMatrix(cars):
    A = []
    b = []
    for car in cars:
        row = [car.getAge(), car.getTechCheck()]
        A.append(row)
        b.append(car.getCondition())
    b = np.array(b)
    A = np.array(A)
    return A, b
