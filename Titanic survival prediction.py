
# coding: utf-8

# In[643]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.filterwarnings('ignore')


# In[644]:


df = pd.read_csv('/home/stephen/Classes/Big Data Analytics/Titanic survival prediction/train.csv')
df.head(3)


# In[645]:


df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


# In[646]:


df.isna().sum()
AgeMedian = df['Age'].median()
df['Age'].replace(np.nan, AgeMedian, inplace=True)
df['Embarked'].value_counts()
df['Embarked'].replace(np.nan, 'S', inplace=True)
df.isna().sum()


# In[647]:


df['Female'] = np.where((df['Sex'] == 'female'), 1, 0)
df = df.drop('Sex', axis = 1)
df.head(3)


# In[648]:


df['Embarked'].value_counts()


# In[649]:


df['Embarked_S'] = np.where((df['Embarked'] == 'S'), 1, 0)
df['Embarked_C'] = np.where((df['Embarked'] == 'C'), 1, 0)
df['Embarked_Q'] = np.where((df['Embarked'] == 'Q'), 1, 0)
df = df.drop('Embarked', axis = 1)
df.head(10)


# In[650]:


df['Parch'].value_counts()


# In[651]:


df['Parch0'] = np.where((df['Parch'] == 0), 1, 0)
df['Parch1'] = np.where((df['Parch'] == 1), 1, 0)
df['Parch2'] = np.where((df['Parch'] == 2), 1, 0)
df['Parch3'] = np.where((df['Parch'] == 3), 1, 0)
df['Parch4'] = np.where((df['Parch'] == 4), 1, 0)
df['Parch5'] = np.where((df['Parch'] == 5), 1, 0)
df['Parch6'] = np.where((df['Parch'] == 6), 1, 0)
df = df.drop('Parch', axis = 1)
df.head(10)


# In[652]:


df['Pclass'].value_counts()


# In[653]:


df['Pclass1'] = np.where((df['Pclass'] == 1), 1, 0)
df['Pclass2'] = np.where((df['Pclass'] == 2), 1, 0)
df['Pclass3'] = np.where((df['Pclass'] == 3), 1, 0)
df = df.drop('Pclass', axis = 1)
df.head(10)


# In[654]:


df['SibSp'].value_counts()


# In[655]:


df['SibSp0'] = np.where((df['SibSp'] == 0), 1, 0)
df['SibSp1'] = np.where((df['SibSp'] == 1), 1, 0)
df['SibSp2'] = np.where((df['SibSp'] == 2), 1, 0)
df['SibSp3'] = np.where((df['SibSp'] == 3), 1, 0)
df['SibSp4'] = np.where((df['SibSp'] == 4), 1, 0)
df['SibSp5'] = np.where((df['SibSp'] == 5), 1, 0)
df = df.drop(('SibSp'), axis = 1)
df.head(10)


# df['Cabin'].value_counts()
# df['Cabin'] = df['Cabin'].astype(str)
# if (df['Cabin'].str.contains("A")): 
#     df['FirstClass'] = np.where((df['Cabin'] == 0), 1, 0)
# df = df.drop(('Cabin'), axis = 1)
# df.head(10)
# df.info()

# In[656]:


df.Cabin = df.Cabin.astype(str)
df.Cabin = df.Cabin.map(lambda x: x[0])
cabin_dummies = pd.get_dummies(df.Cabin, prefix="Cabin")
df = pd.concat([df, cabin_dummies], axis=1)
df.drop(['Cabin'],axis=1)


# In[657]:


Survived =len(df[df['Survived'] == 1])
Died = len(df[df['Survived']== 0])

plt.figure(figsize=(8,6))

labels = 'Survived','Died'
sizes = [Survived, Died]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Female =len(df[df['Female'] == 1])
Male = len(df[df['Female']== 0])

plt.figure(figsize=(8,6))

labels = 'Female','Male'
sizes = [Female, Male]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Embarked_S'] == 1])
No = len(df[df['Embarked_S']== 0])

plt.figure(figsize=(8,6))

labels = 'S','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Embarked_C'] == 1])
No = len(df[df['Embarked_C']== 0])

plt.figure(figsize=(8,6))

labels = 'C','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Embarked_Q'] == 1])
No = len(df[df['Embarked_Q']== 0])

plt.figure(figsize=(8,6))

labels = 'Q','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Parch0'] == 1])
No = len(df[df['Parch0']== 0])

plt.figure(figsize=(8,6))

labels = 'Parch_0','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Parch1'] == 1])
No = len(df[df['Parch1']== 0])

plt.figure(figsize=(8,6))

labels = 'Parch_1','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Parch2'] == 1])
No = len(df[df['Parch2']== 0])

plt.figure(figsize=(8,6))

labels = 'Parch_2','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Parch3'] == 1])
No = len(df[df['Parch3']== 0])

plt.figure(figsize=(8,6))

labels = 'Parch_3','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Parch4'] == 1])
No = len(df[df['Parch4']== 0])

plt.figure(figsize=(8,6))

labels = 'Parch_4','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Parch5'] == 1])
No = len(df[df['Parch5']== 0])

plt.figure(figsize=(8,6))

labels = 'Parch_5','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Parch6'] == 1])
No = len(df[df['Parch6']== 0])

plt.figure(figsize=(8,6))

labels = 'Parch_6','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Pclass1'] == 1])
No = len(df[df['Pclass1']== 0])

plt.figure(figsize=(8,6))

labels = 'Pclass1','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Pclass2'] == 1])
No = len(df[df['Pclass2']== 0])

plt.figure(figsize=(8,6))

labels = 'Pclass2','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['Pclass3'] == 1])
No = len(df[df['Pclass3']== 0])

plt.figure(figsize=(8,6))

labels = 'Pclass3','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['SibSp0'] == 1])
No = len(df[df['SibSp0']== 0])

plt.figure(figsize=(8,6))

labels = 'SibSp0','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['SibSp1'] == 1])
No = len(df[df['SibSp1']== 0])

plt.figure(figsize=(8,6))

labels = 'SibSp1','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['SibSp2'] == 1])
No = len(df[df['SibSp2']== 0])

plt.figure(figsize=(8,6))

labels = 'SibSp2','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['SibSp3'] == 1])
No = len(df[df['SibSp3']== 0])

plt.figure(figsize=(8,6))

labels = 'SibSp3','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['SibSp4'] == 1])
No = len(df[df['SibSp4']== 0])

plt.figure(figsize=(8,6))

labels = 'SibSp4','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()

Yes =len(df[df['SibSp5'] == 1])
No = len(df[df['SibSp5']== 0])

plt.figure(figsize=(8,6))

labels = 'SibSp5','Other'
sizes = [Yes, No]
colors = ['lavender', 'turquoise']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# In[658]:


plt.hist(df['Age'],bins=12)


# In[659]:


plt.hist(df['Fare'])


# In[660]:


testdf = pd.read_csv('/home/stephen/Classes/Big Data Analytics/Titanic survival prediction/test.csv')

testdf.isna().sum()


# In[661]:


AgeMedian = df['Age'].median()
testdf['Age'].replace(np.nan, AgeMedian, inplace=True)
testdf['Embarked'].value_counts()
testdf['Embarked'].replace(np.nan, 'S', inplace=True)
FareMedian = df['Fare'].median()
testdf['Fare'].replace(np.nan, FareMedian, inplace=True)
testdf.isna().sum()


# In[662]:


testdf['Female'] = np.where((testdf['Sex'] == 'female'), 1, 0)
testdf = testdf.drop('Sex', axis = 1)
testdf = testdf.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
testdf['Embarked_S'] = np.where((testdf['Embarked'] == 'S'), 1, 0)
testdf['Embarked_C'] = np.where((testdf['Embarked'] == 'C'), 1, 0)
testdf['Embarked_Q'] = np.where((testdf['Embarked'] == 'Q'), 1, 0)
testdf = testdf.drop('Embarked', axis = 1)
testdf['Pclass1'] = np.where((testdf['Pclass'] == 1), 1, 0)
testdf['Pclass2'] = np.where((testdf['Pclass'] == 2), 1, 0)
testdf['Pclass3'] = np.where((testdf['Pclass'] == 3), 1, 0)
testdf = testdf.drop('Pclass', axis = 1)
testdf['Parch0'] = np.where((testdf['Parch'] == 0), 1, 0)
testdf['Parch1'] = np.where((testdf['Parch'] == 1), 1, 0)
testdf['Parch2'] = np.where((testdf['Parch'] == 2), 1, 0)
testdf['Parch3'] = np.where((testdf['Parch'] == 3), 1, 0)
testdf['Parch4'] = np.where((testdf['Parch'] == 4), 1, 0)
testdf['Parch5'] = np.where((testdf['Parch'] == 5), 1, 0)
testdf['Parch6'] = np.where((testdf['Parch'] == 6), 1, 0)
testdf = testdf.drop(('Parch'), axis = 1)
testdf['SibSp0'] = np.where((testdf['SibSp'] == 0), 1, 0)
testdf['SibSp1'] = np.where((testdf['SibSp'] == 1), 1, 0)
testdf['SibSp2'] = np.where((testdf['SibSp'] == 2), 1, 0)
testdf['SibSp3'] = np.where((testdf['SibSp'] == 3), 1, 0)
testdf['SibSp4'] = np.where((testdf['SibSp'] == 4), 1, 0)
testdf['SibSp5'] = np.where((testdf['SibSp'] == 5), 1, 0)
testdf = testdf.drop(('SibSp'), axis = 1)
testdf.Cabin = testdf.Cabin.astype(str)
testdf.Cabin = testdf.Cabin.map(lambda x: x[0])
cabin_dummies = pd.get_dummies(testdf.Cabin, prefix="Cabin")
testdf = pd.concat([testdf, cabin_dummies], axis=1)
testdf = testdf.drop(('Cabin'),axis=1)


# In[663]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

df.info()
df = df.drop(('Cabin'),axis=1)
df = df.drop(('Cabin_T'),axis=1)
X_train = df.drop(['Survived'], axis=1)
y_train = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=9)


# In[664]:


#1) Logistic Regression
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
print('Classification Report: \n', classification_report(y_test,log_predicted))

sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")


# In[665]:


#2) LinearSVC classifier
SVC = LinearSVC()
# Train the model using the training sets and check score
SVC.fit(X_train, y_train)
#Predict Output
SVC_predicted= SVC.predict(X_test)

SVC_score = round(SVC.score(X_train, y_train) * 100, 2)
SVC_score_test = round(SVC.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('SVC Regression Training Score: \n', SVC_score)
print('SVC Regression Test Score: \n', SVC_score_test)
print('Accuracy: \n', accuracy_score(y_test,SVC_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,SVC_predicted))
print('Classification Report: \n', classification_report(y_test,SVC_predicted))

sns.heatmap(confusion_matrix(y_test,SVC_predicted),annot=True,fmt="d")


# In[666]:


#3) Gradient Boosting Classifier
Gradient = GradientBoostingClassifier()
# Train the model using the training sets and check score
Gradient.fit(X_train, y_train)
#Predict Output
Gradient_predicted= Gradient.predict(X_test)

Gradient_score = round(Gradient.score(X_train, y_train) * 100, 2)
Gradient_score_test = round(Gradient.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Gradient Boosting Training Score: \n', Gradient_score)
print('Gradient Boosting Test Score: \n', Gradient_score_test)
print('Accuracy: \n', accuracy_score(y_test,Gradient_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,Gradient_predicted))
print('Classification Report: \n', classification_report(y_test,Gradient_predicted))

sns.heatmap(confusion_matrix(y_test,Gradient_predicted),annot=True,fmt="d")


# In[667]:


#4) Decision Tree Classifier
Decider = DecisionTreeClassifier()
# Train the model using the training sets and check score
Decider.fit(X_train, y_train)
#Predict Output
Decider_predicted= Decider.predict(X_test)

Decider_score = round(Decider.score(X_train, y_train) * 100, 2)
Decider_score_test = round(Decider.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Decision Tree Training Score: \n', Decider_score)
print('Decision Tree Test Score: \n', Decider_score_test)
print('Accuracy: \n', accuracy_score(y_test,Decider_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,Decider_predicted))
print('Classification Report: \n', classification_report(y_test,Decider_predicted))

sns.heatmap(confusion_matrix(y_test,Decider_predicted),annot=True,fmt="d")


# In[668]:


#5) Random Forest Classifier
Random = RandomForestClassifier()
# Train the model using the training sets and check score
Random.fit(X_train, y_train)
#Predict Output
Random_predicted = Random.predict(X_test)

Random_score = round(Random.score(X_train, y_train) * 100, 2)
Random_score_test = round(Random.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Random Forest Training Score: \n', Random_score)
print('Random Forest Test Score: \n', Random_score_test)
print('Accuracy: \n', accuracy_score(y_test,Random_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,Random_predicted))
print('Classification Report: \n', classification_report(y_test,Random_predicted))

sns.heatmap(confusion_matrix(y_test,Random_predicted),annot=True,fmt="d")


# In[669]:


X_train = df.drop(['Survived'], axis=1)
y_train = df['Survived']
logreg.fit(X_train, y_train)

Gradient_predicted = Gradient.predict(testdf)


# In[670]:


testdf = pd.read_csv('/home/stephen/Classes/Big Data Analytics/Titanic survival prediction/test.csv')

PassengerId = testdf['PassengerId'].tolist()
PassengerId = ['PassengerId'] + PassengerId
Gradient_predicted = Gradient_predicted.tolist()
Gradient_predicted = ['Survived'] + Gradient_predicted
Submission = np.column_stack((PassengerId, Gradient_predicted))


# In[671]:


len(Submission)


# In[672]:


pd.DataFrame(Submission).to_csv("/home/stephen/Classes/Big Data Analytics/Titanic survival prediction/Submission.csv", header = False)

