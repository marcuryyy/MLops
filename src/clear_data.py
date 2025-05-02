import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

input_path = "data/diamonds.csv"
output_path = "data/diamonds_clean.csv"

df = pd.read_csv(input_path)


price_col = df['price']
df = df.drop(['price'], axis=1)  # Удалим колонку с ценой, т.к. это наша целевая переменная

# В колонках пропущенных значений нет, а это значит, можем сразу переходить к кодированию данных
cat_cols = df.select_dtypes(include=['object', 'category']).columns
num_cols = df.select_dtypes(include=['number']).columns

# Кодировать будем с помощью OneHotEncoder, дропая первую колонку параметром drop = 'first'
ohe = OneHotEncoder(drop='first', sparse_output=False)

encoded_data = ohe.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(cat_cols))

df = df.drop(cat_cols, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# Для колонок, которые содержали числовые значения, применим StandardScaler, чтобы не было больших отличий между
# данными (в т.ч. выбросы).
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df['price'] = price_col # вернем колонку с ценой
df = df.drop_duplicates()
df = df.dropna()
print(df.isnull().sum())
df.to_csv(output_path, index=False)
