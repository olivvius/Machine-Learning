import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, recall_score, accuracy_score, precision_score,recall_score, f1_score,PrecisionRecallDisplay
from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostClassifier, Pool


#Preprocessing et EDA
data=pd.read_csv('heart.csv')
data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)
data['ExerciseAngina'].replace(['N', 'Y'], [0, 1], inplace=True)
data.Cholesterol.replace(['0'], data.Cholesterol.median(), inplace=True)

data2=pd.get_dummies(data, columns=['ChestPainType','RestingECG','ST_Slope'])
data2['target']=data2.pop('HeartDisease')
S=data2.shape
#sns.pairplot(data2, hue='HeartDisease')
#sns.swarmplot(y="Age", x="Sex", hue="HeartDisease", data=df, s=4, palette=colors)
plt.figure(figsize=(15,15))
sns.heatmap(data2.corr(),cmap='coolwarm', annot=True, vmin=-1, vmax=1)
#sns.clustermap(data2.corr())

#selection des variables numériques
num= data2.drop(columns='target').select_dtypes(include=['float64','int64'])
#affichage des outliers
# plt.figure(figsize=(20, 10))
# for i, col in enumerate(num):
#     ax = plt.subplot(2, 3, i+1)
#     sns.boxplot(x=num[col])
# plt.tight_layout()


data_n=pd.DataFrame(MinMaxScaler().fit_transform(data2),index=data2.index, columns=data2.columns)
trainset, testset = train_test_split(data_n, test_size=0.2, random_state=0)
# print(trainset['HeartDisease'].value_counts())
# print(testset['HeartDisease'].value_counts())

y_train=trainset.target
X_train=trainset.drop('target',axis=1)
y_test=testset.target
X_test=testset.drop('target',axis=1)


#Test avec linearSVC
model=LinearSVC(max_iter=3000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
coef=model.coef_

cm=confusion_matrix(y_test, y_pred)
cl=classification_report(y_test, y_pred)

#Test avec MultinomialNB
model2=MultinomialNB()
model2.fit(X_train, y_train)
y_pred2 = model.predict(X_test)

cm2=confusion_matrix(y_test, y_pred2)
cl2=classification_report(y_test, y_pred2)

#Test avec CatBoostClassifier
model3=CatBoostClassifier()

paramgrid =  {'n_estimators':[250,300,400,500],'max_depth':[5,6,7,8]}
grid=GridSearchCV(estimator=model3,param_grid=paramgrid, scoring ='accuracy', cv = 5)

grid.fit(X_train, y_train)
best=grid.best_params_
bestmodel=grid.best_estimator_
y_pred3 = bestmodel.predict(X_test)

cm3=confusion_matrix(y_test, y_pred3)
cl3=classification_report(y_test, y_pred3)


