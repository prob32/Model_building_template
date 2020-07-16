import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

df = pd.read_csv('tc20171021.csv')
df = df.drop(columns=['Id'])

def counts_viz (make,threshold):
    make_more_than_twenty_thousand = df.groupby(make).filter(lambda x: len(x) > threshold)
    ax = sns.countplot(x=make, data=make_more_than_twenty_thousand)
    plt.show()

#counts_viz('Make',20000)

def categorical_coding(column_name,dataframe):
    dataframe[column_name] = dataframe[column_name].astype('category')
    dataframe[column_name] = dataframe[column_name].cat.codes

#categorical_coding('Make',df)
#categorical_coding('Model',df)

corr = df.corr()

def heatmap ():
    ax = sns.heatmap(
        corr, annot=True,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()

def reg_sp(x,y):
    dfSample = df.sample(1000)
    ax = sns.regplot(x=x, y=y, data=dfSample, x_jitter=.1)
    plt.show()




def make_mileage (make,threshold):
    mileage_threshold = df.groupby(make).filter(lambda x: len(x) > threshold)
    ax = sns.barplot(x=make, y="Mileage", data=mileage_threshold)
    plt.show()

make_mileage('Make',8000)