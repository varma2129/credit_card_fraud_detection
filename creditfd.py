import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression


#creating a dataframe to work on the data available
df=pd.read_csv('creditcard.csv')
df.head()
df.describe()


#creating a smaller dataframe for time and ammount and scaling them
rbs = RobustScaler()
df_small = df[['Time','Amount']]
df_small = pd.DataFrame(rbs.fit_transform(df_small))
df_small.columns = ['scaled_time','scaled_amount']
df = pd.concat([df,df_small],axis=1)


#drop the unscaled columns 
df.drop(['Time','Amount'],axis=1,inplace=True)
df.head()

#using sns we visualize the data in graphical form
df['Class'].value_counts()
sns.countplot(df['Class'])

#taking out fraud and non fraud out of the tables
non_fraud = df[df['Class']==0]
fraud = df[df['Class']==1]

non_fraud = non_fraud.sample(frac=1)


#"n" to be replaced with the total instances that are fraud
non_fraud = non_fraud[:n]

#now defining a new dataframe with fraud and non fraud transactions in it
new_df = pd.concat([non_fraud,fraud])
new_df = new_df.sample(frac=1)
new_df['Class'].value_counts()
sns.countplot(new_df['Class'])

X = new_df.drop('Class',axis=1)
Y = new_df['Class']


#splitting the data in 80-20% for training and testing respectively
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=101)

#using logistic regression to train our model in detection
lr = LogisticRegression()
lr.fit(X_train,Y_train)

pred = lr.predict(X_test)

print(classification_report(Y_test,pred))
print('\n')
print(confusion_matrix(Y_test,pred))
print('\n')
print('accuracy of model is --> ',round(accuracy_score(Y_test,pred)*100,2))