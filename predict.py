import pandas as pd
from PIL import Image
import scipy.misc
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs


swanmap = pd.read_csv('/Volumes/DataDisk/Downloads/data.csv')

fininsh = [[0]]
fininsh2 = [[0]]
print swanmap.info()

for i in range(1,70):
    y = swanmap[str(i)]
    feature_matrix = swanmap.drop([str(i)], axis=1)

    X = pd.get_dummies(feature_matrix, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=None)
    lr = LinearRegression()

    #training
    lr.fit(X_train, y_train)

    #prediction
    diabetes_y_pred = lr.predict(X_test)
    fininsh = np.concatenate((fininsh,[[[255-diabetes_y_pred[1]]]]),axis=0)
    fininsh2 = np.concatenate((fininsh2,[[255-diabetes_y_pred[10]]]),axis=0)
    
fininsh3 = np.hstack((fininsh,fininsh2))


#print finish4
#print finish3

#fininsh = np.array([[diabetes_y_pred[1]]]+[[diabetes_y_pred2[1]]]+[[diabetes_y_pred2[3]]]+[[diabetes_y_pred2[3]]]+[[diabetes_y_pred5[1]]])
#plt.plot(fininsh)
#plt.ylabel('some numbers')
#plt.show()

scipy.misc.imsave('outfile.jpg', fininsh)
scipy.misc.imsave('outfile2.jpg', fininsh2)
scipy.misc.imsave('outfile3.jpg', fininsh3)
