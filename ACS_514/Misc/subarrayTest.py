import numpy as np

thing = np.matrix([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

sub = thing[2:,2:]

print(sub)