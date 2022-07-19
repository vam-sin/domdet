import os
import numpy as np
import matplotlib.pyplot as plt

def make_heatmap(pae):
    plt.imshow(pae, cmap='hot', interpolation='nearest')
    plt.show()



if __name__=="__main__":
    for f in os.listdir('../features/r5_test_pae'):
        try:
            r5 = np.load('../features/r5_test_pae/' + f)
            r1 = np.load('../features/test_pae/' + f)
            bp = True
        except:
            pass