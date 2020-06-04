import numpy as np

# NUMPY CAN MULTIPLY ARRAYS
a = [1, 3, 5]
b = [1, 2, 3]

try:
    print(a*b)
except Exception as e:
    print(e)

a1 = np.array([1, 3, 5])
b1 = np.array([1, 2, 3])

try:
    li = a1*b1
    print(type(li), '\n')
except Exception as e:
    print(e)

# NUMPY HAS MULTIDIMENSIONAL ARRAYS

ar0 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
print(ar0, '\n')

# np.random
ar1 = np.random.randint(0, 1000000000, size=(4, 4))
print(ar1, '\n')

# np.random with 3d array
ar2 = np.random.randint(0, 100, size=(4, 4, 4))
print(ar2, '\n')

# np.repeat
ar_t = np.array([[0, 10, 100, 1000]])
ar3 = np.repeat(ar_t, 5, axis=0)
print(ar3, '\n')

# art shit
op = np.ones((11, 11))
zer = np.zeros((9, 9))
zer[4, 4] = 9
op[1:-1, 1:-1] = zer
print(op, '\n')

# copying can be dangerous
ar4 = np.array([1, 2, 3])
ar5 = ar4
ar5[0] = '69'
print('ar4: ', ar4)
print('ar5: ', ar5, '\n')

# .copy
ar6 = np.array([1, 2, 3])
ar7 = ar4.copy()
ar7[0] = '69'
print('ar6: ', ar6)
print('ar7: ', ar7, '\n')

# maths
ar8 = np.array([[1, 2, 3, 4, 5]])
print(ar8)
print(ar8+1)
print(ar8-1)
print(ar8*2)
print(ar8/2)
print(ar8**2)
print(ar8%2, '\n')
