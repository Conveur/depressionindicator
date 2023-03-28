import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class Model:

    def __init__(self):
        # Reading the dataset into a dataframe
        self.name = ''
        df = pd.read_csv('depressionDataset.csv', sep=',', engine='python')
        df = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'class']]

        # Filling missing data
        df['q1'] = df['q1'].fillna(df['q1'].mode()[0])
        df['q2'] = df['q2'].fillna(df['q2'].mode()[0])
        df['q3'] = df['q3'].fillna(df['q3'].mode()[0])
        df['q4'] = df['q4'].fillna(df['q4'].mode()[0])
        df['q5'] = df['q5'].fillna(df['q5'].mode()[0])
        df['q6'] = df['q6'].fillna(df['q6'].mode()[0])
        df['q7'] = df['q7'].fillna(df['q7'].mode()[0])
        df['q8'] = df['q8'].fillna(df['q8'].mode()[0])
        df['q9'] = df['q9'].fillna(df['q9'].mode()[0])
        df['q10'] = df['q10'].fillna(df['q10'].mode()[0])
        df['class'] = df['class'].fillna(df['class'].mode()[0])

        self.split_data(df)

    def split_data(self,df):
        # Separating the dataframe into X and y data
        X = df.values
        y = df['class'].values

        # Deleting 'class' column from X
        X = np.delete(X,10,axis=1)

        # Splitting the data into 70% for training and 30% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def RandomForestclassifier(self):
        # Building Random Forest classifier
        self.name = 'Random Forest Classifier'
        classifier = RandomForestClassifier()
        return classifier.fit(self.X_train, self.y_train)

    def GradientBoostingclassifier(self):
        # Building Gradient Boosting classifier
        self.name = 'Gradient Boosting Classifier'
        classifier = GradientBoostingClassifier(n_estimators=100)
        return classifier.fit(self.X_train, self.y_train)

    def SVMclassifier(self):
        # Building SVM classifier
        self.name = 'SVM Classifier'
        classifier = SVC()
        return classifier.fit(self.X_train, self.y_train)
    
    def accuracy(self,model):
        # Calculating the accuracy of a classifier and producing a confusion matrix
        predictions = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions)
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        print(f"{self.name} has accuracy of {accuracy *100} % ")

if __name__ == '__main__':
    model = Model()
    model.accuracy(model.RandomForestclassifier())
    model.accuracy(model.GradientBoostingclassifier())
    model.accuracy(model.SVMclassifier()) 



