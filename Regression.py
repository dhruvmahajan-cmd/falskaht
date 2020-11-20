import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model
import pickle

db = pd.read_csv('Aht_Updated.csv', sep=',')

X = np.array(db.drop(['AWT','ATT'], axis=1))
y = np.array(db['AWT'])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# best = 0

# for d in range(1000):
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


#     linear = linear_model.LinearRegression()
#     linear.fit(X_train, y_train)


#     accuracy = linear.score(X_test, y_test)

#     print(accuracy)
    
#     if accuracy > best:
#         best = accuracy
#         print('The best is : ' + str(best))
#         with open('Aht_Model.pickle', 'wb') as f:
#             pickle.dump(linear, f)
            

linear = pickle.load(open('Aht_Model.pickle', 'rb'))

print(linear.score(X_test, y_test))


predictions = linear.predict(X_test)

print(predictions)
# for x in range(len(predictions)):
# # if (predictions[x] - y_test[x]) < 5 and (predictions[x] - y_test[x]) > 0:
#     print(predictions[x], y_test[x],(predictions[x] - y_test[x]), X_test[x] )
