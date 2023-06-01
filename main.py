import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE

df=pd.read_csv(r"the train path")

#Dropping duplicate values
df.drop_duplicates(inplace=True)

#Removing irrelevant data
list_of_undergrads=["1th","2th","3th","4th","5th","6th","7th","8th","9th","10th","11th","12th"]
for x in df.index:
     if df.loc[x,"education"].strip() in list_of_undergrads and (df.loc[x,"marital-status"].strip()!= "Never-married" or df.loc[x,"salary"].strip()==">50K"):
         df.drop(x,inplace=True)

#Dealing with Null values
df=df.replace(' ?', np.NaN)
#Getting the percentage of Null values
#percent_missing = (df.isna().sum() * 100) / len(df)
#Dropping Null values
df.dropna(how='any',inplace=True)


# Encode Categorical Columns
categ = ['work-class','education','marital-status','position','relationship','race','sex','native-country','salary']
# Encode Categorical Columns
le = LabelEncoder()
df[categ]=df[categ].apply(le.fit_transform)


# Apply correlation
corr=df.corr().round(2)
filteredDf = corr[((corr >= 0.2) | (corr <= -0.2)) & (corr !=1.000)]
# #Correlation of all features with the target column
# top_feature=corr.index[abs(corr['salary']>0.2) ]
# top_corr = df[top_feature].corr()
# sns.heatmap(top_corr,cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True)

plt.subplots(figsize=(12, 8))
sns.heatmap(filteredDf,cmap=sns.diverging_palette(220, 10, as_cmap=True),annot=True)
plt.show()

#Removing unneed columns
columns=['work-class','work-fnl','education','position','race','capital-loss','native-country','relationship']
df.drop(columns, inplace=True, axis=1)

#Spliting the dataset into X and Y
X=df.iloc[:,0:6] #Features
Y=df['salary'] #Label

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=109) # 80% training and 20% test


# Solving imbalanced classes problem using oversampling(smote)
smote = SMOTE()
X, Y = smote.fit_resample(X_train, y_train)


#Create a SVM Classifier
clf = svm.SVC(kernel='rbf',C=100,gamma=0.01,max_iter=100000000) # rbf Kernel
#Train the model using the training sets
model=clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
#Model Accuracy
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Create a logistic classifier
lr1 = LogisticRegression(C=16.768329368110066, penalty='l2',max_iter=1000000000)
#Train the model using the training sets
model=lr1.fit(X_train, y_train)
# Model Accuracy
print('Logistic classifier Accuracy: {:.3f}'.format(lr1.score(X_test,y_test)))

#Apply grid search on KNN
grid_params = { 'n_neighbors' : [5,7,9,11,13,15,20,100],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
g_res = gs.fit(X_train, y_train)
print(g_res.best_params_)

#Create KNN Classifier
KNN = KNeighborsClassifier(n_neighbors=100,metric='manhattan',weights='distance')
#Cross Validation
print("cross:",cross_val_score(KNN, X, Y, cv=3 , scoring='accuracy').mean())
#Train the model using the training sets
model=KNN.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = KNN.predict(X_test)
# Model Accuracy
print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred))



#Create a RandomForest Classifier
rfc=RandomForestClassifier(n_estimators=400,max_depth=8)
#Train the model using the training sets y_pred=clf.predict(X_test)
model=rfc.fit(X_train,y_train)
#Predict the response for test dataset
y_pred=rfc.predict(X_test)
#Model Accuracy
print("Random forest Accuracy:",metrics.accuracy_score(y_test, y_pred))



#Classes frequency bar graph
classes = Y.values
unique, counts = np.unique(classes, return_counts=True)
plt.bar(unique,counts)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

############################################################testing phase##############################################################
df2=pd.read_csv(r"the test path")

#Dealing with Null values
df2=df2.replace(' ?', np.NaN)
#Getting the percentage of Null values
# percent_missing = (df2.isna().sum() * 100) / len(df2)

#Replacing Null values with the mode
df2['workclass'] = df2['workclass'].fillna(df2['workclass'].mode()[0])
df2['native-country'] = df2['native-country'].fillna(df2['native-country'].mode()[0])
df2['occupation'] = df2['occupation'].fillna(df2['occupation'].mode()[0])


# Encode Categorical Columns
categ2 = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
# Encode Categorical Columns
le2 = LabelEncoder()
df2[categ2] = df2[categ2].apply(le2.fit_transform)

#Removing unneed columns
columns=['workclass','fnlwgt','occupation','education','relationship','race','capital-loss','native-country']
df2.drop(columns, inplace=True, axis=1)

#Features
X1=df2.iloc[:,0:6]

#Predict the response for test dataset
Y_pred = model.predict(X1)
Y_pred=le.inverse_transform(Y_pred)
df3 = pd.DataFrame({'salary':Y_pred})
df3.to_csv(r"Your output csv file Path")