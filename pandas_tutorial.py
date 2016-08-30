import csv as csv
import numpy as np
import pandas as pd

def pandas_tutorial():
    
    csv_file_object = csv.reader(open('csv/train.csv', 'r'))
    header = csv_file_object.__next__()
    data=[]

    for row in csv_file_object:
        data.append(row)

    data = np.array(data) 
    print(data)

    s = pd.Series([1,3,5,np.nan,6,8])
    print(s)

    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    print(dates, '\n', df)

    df2 = pd.DataFrame({'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo' })
    print(df2)

    print(df.head())
    print(df.index)
    print(df.columns)
    print(df.describe())
    print(df2.describe())

    print(df['A'], '\n')
    print(df[0:3], '\n')

    print(df.loc('A'), '\n')
    print(df.iloc(1:3), '\n')
    
if __name__ == '__main__':
    pandas_tutorial()
