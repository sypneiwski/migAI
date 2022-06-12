import numpy as np
def test(array):
    return chr(int(np.average(array)) % (ord('z') - ord('a')) + ord('a'))