import csv as csv
import numpy as np
import pandas as pd

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

def pandas_smart_kaggle():    
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv('csv/train.csv', header=0)

    print(df)
    print(df.describe())
    print(df.loc[15,'Age'])
    print(df.info())

    print(df[['Age', 'Cabin', 'Fare']])
    print('mean : ', df['Age'].mean())
    print('median : ', df['Age'].median())

    print(df[ (df['Age'] > 60) & (df['Survived'] == 1)]  )

    df['Gender']   = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    print(df['Embarked'].head(10))
    df['Embarked'] = df['Embarked'].map( lambda x: (1 if x == 'S' else 0) ).astype(int)
#    print(df['Gender'])
    print(df['Embarked'].head(10))
    
if __name__ == '__main__':
#    pandas_kaggle()
    pandas_smart_kaggle()
