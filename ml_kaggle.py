import csv as csv
import numpy as np
import pandas as pd

import sklearn.ensemble as ensemble
import matplotlib.pyplot as plt

def pandas_kaggle():
    
    csv_file_object = csv.reader(open('csv/train.csv', 'r'))
    header = csv_file_object.__next__()
    data=[]

    for row in csv_file_object:
        data.append(row)

    data = np.array(data) 
    print(data)

    df = pd.DataFrame(data, columns=header)
    print(df)
    print(df.describe())
    print(df.loc[15,'Age'])
def cleanDataFrame(df) :
    df['Gender']   = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked.loc[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
        # Again convert all Embarked strings to int
        df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
            
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & \
                                  (df['Pclass'] == j+1)]['Age'].dropna().median()
    
    df['AgeFill'] = df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                    'AgeFill'] = median_ages[i,j]
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass

    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked' , 'Age'], axis=1)
    print(df)
    print(df.describe())
    print(df.info())
    
def pandas_smart_kaggle():    
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv('csv/train.csv', header=0)

    cleanDataFrame(df)

    train_data = df.values
    print(train_data)
    
    test_df = pd.read_csv('csv/test.csv', header=0)
    

    try :
        cleanDataFrame(test_df)
    except :
        print("failed to clean data frame ! exiting")
        exit()

    test_data = test_df.values
    
    randomForest = ensemble.RandomForestClassifier()    
    randomForest.fit(train_data[0::,1::] , train_data[0::,0] )

    print(randomForest.predict(test_data))
    
    # plt.figure()
    # df['Age'].hist()
    # plt.figure()
    # df['AgeFill'].hist()
    # plt.show()    

    
if __name__ == '__main__':
#    pandas_kaggle()
    pandas_smart_kaggle()
