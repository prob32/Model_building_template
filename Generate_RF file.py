import pandas as pd

df = pd.read_csv('cars_500k.csv')

df= df.drop(columns=['Unnamed: 9','Vin','City'])
df = df.sample(50000)

df_list = list(df.columns)
print(df_list)

final_data = pd.get_dummies(data=df, columns=['Year', 'Make','Model','State'])
final_data.to_csv('mini_model_load.csv')

df_list = list(final_data.columns)
print(df_list)