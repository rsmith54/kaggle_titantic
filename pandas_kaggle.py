import csv as csv
import numpy as np
import pandas as pd

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
    
if __name__ == '__main__':
    pandas_kaggle()
    pandas_smart_kaggle()
