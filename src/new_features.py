import pandas as pd

input_path = "data/diamonds_clean.csv"
temp_data_path = "data/diamonds.csv"
output_path = "data/diamonds_features.csv"

df = pd.read_csv(input_path)
temp_df = pd.read_csv(temp_data_path)
df['volume'] = df['x'] * df['y'] * df['z']  # Добавим новый признак - объем
df['median_carat_by_color'] = temp_df.groupby('color')['carat'].transform('median') # Добавим новым признак - среднее значение карат по цвету
df.to_csv(output_path, index=False)