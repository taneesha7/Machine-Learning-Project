import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
#IMPORTING DATASET
data=pd.read_csv(r"Processed_Model_Data1.csv")
data.head()
#SELECTING COLUMNS(ATTRIBUTES)
data=data.iloc[:,[0,1,2,3,4,5,6,7,8]] 
print(data)
#PRINTING DATATYPES
data.dtypes
#LOOKING FOR NULL VALUES
data.isnull().sum()
#VISUALIZING THE VALUES IN A MATRIX
sns.heatmap(data.corr(),annot= True)
#LABEL ENCODING THE ATTRIBUTES TO NUMERIC VALUES
le=LabelEncoder()
data['Sex']=le.fit_transform(data.Sex)
data['Outcome']=le.fit_transform(data.Outcome)
data
#FIND CORRELATION OF ALL DATAFRAMES
data.corr()
sns.heatmap(data.corr(),annot=True)
#COUNT NO OF OUTCOMES OF DIFFERENT TYPES 
data['Outcome'].value_counts()
print("Gender distribution:\n",data['Sex'].value_counts(),"\n")
print("Outcome distribution:\n",data['Outcome'].value_counts(),"\n")
#SPLITTING DATASET INTO TEST-TRAIN SET
X,Y=data.iloc[:,0:7],data.iloc[:,8]
train_x,test_x,train_y,test_y= train_test_split(X,Y,test_size=0.2,random_state=50)
print("shape of train_x=",train_x.shape)
print("shape of test_x=",test_x.shape)
print("shape of train_y=",train_y.shape)
print("shape of test_y=",test_y.shape)
#LOGISTIC REGRESSION
LR = LogisticRegression(max_iter=1000)
LR.fit(train_x,train_y)
pred= LR.predict(test_x)
print("Confusion matrix of Logisitc Regression Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Logisitc Regression Model         \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Logisitc Regression Model     \t",metrics.recall_score(test_y,pred))
print("Precision Score of Logisitc Regression Model  \t",metrics.precision_score(test_y,pred))
print("f1 score of Logisitc Regression Model         \t",metrics.f1_score(test_y,pred))
#RANDOM FOREST
rf=RandomForestClassifier()
rf.fit(train_x,train_y)
pred=rf.predict(test_x)
print("Confusion matrix of Random Forest Classifier Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Random Forest Classifier Model        \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Random Forest Classifier Model    \t",metrics.recall_score(test_y,pred))
print("Precision Score of Random Forest Classifier Model \t",metrics.precision_score(test_y,pred))
print("f1 score of Random Forest Classifier Model        \t",metrics.f1_score(test_y,pred))
#KNN CLASSIFICATION
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(train_x,train_y)
pred= knn.predict(test_x)
print("Confusion matrix of knn Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of knn Model         \t",metrics.accuracy_score(test_y,pred))
print("Recall score of knn Model     \t",metrics.recall_score(test_y,pred))
print("Precision Score of knn Model  \t",metrics.precision_score(test_y,pred))
print("f1 score of knn Model         \t",metrics.f1_score(test_y,pred))
#SVM ALGORITHM
sv = SVC(kernel='linear')
sv.fit(train_x, train_y)
pred= sv.predict(test_x)
print("Confusion matrix of Support vector machine Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Support vector machine Model        \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Support vector machine Model    \t",metrics.recall_score(test_y,pred))
print("Precision Score of Support vector machine Model \t",metrics.precision_score(test_y,pred))
print("f1 score of Support vector machine Model        \t",metrics.f1_score(test_y,pred))
#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(train_x, train_y)
pred= dt.predict(test_x)
print("Confusion matrix of Decision tree Classifier Model\n",metrics.confusion_matrix(test_y,pred))
print("Accuracy of Decision tree Classifier Model        \t",metrics.accuracy_score(test_y,pred))
print("Recall score of Decision tree Classifier Model    \t",metrics.recall_score(test_y,pred))
print("Precision Score of Decision tree Classifier Model \t",metrics.precision_score(test_y,pred))
print("f1 score of Decision tree Classifier Model        \t",metrics.f1_score(test_y,pred))
