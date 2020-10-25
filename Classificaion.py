############################ Importing the Modules #############################
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy as np
############################ Loading the dataset ###############################
dataset = load_iris()

X = dataset.data 
y = dataset.target

features = dataset.data.T

Sepal_length = features[0]
Sepal_width =  features[1]
Petal_length = features[2]
Petal_width =  features[3]

Sepal_length_label = dataset.feature_names[0]
Sepal_width_label =  dataset.feature_names[1]
Petal_length_label = dataset.feature_names[2]
Petal_width_label =  dataset.feature_names[3]
######################## Data Visualization ########################################
# plt.scatter(Sepal_length, Sepal_width, c=dataset.target)
# plt.xlabel(Sepal_length_label)
# plt.ylabel(Sepal_width_label)
# plt.show()
######################## Creating the Model ########################################
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
Model = KNeighborsClassifier(n_neighbors=1)
Model.fit(X_train,y_train)

Sepal_length_Get = float(input('Enter the Sepal Length : '))
Sepal_width_Get = float(input('Enter the Sepal Width : '))
Petal_length_Get = float(input('Enter the Petal Length : '))
Petal_width_Get = float(input('Enter the Petal Width : '))

######################## Predicting the output ######################################
X_new = np.array([[Sepal_length_Get,Sepal_width_Get,Petal_length_Get,Petal_width_Get]])
prediction = Model.predict(X_new)
Percentile = Model.score(X_test,y_test)
Percentile = round(Percentile,2)
print(Percentile*100)
print(prediction)
################################## END ###############################################