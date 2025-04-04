import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def download_dataset():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
    df.to_csv('data.csv', index=False)
    return df


def clear_data(path2df):
    df = pd.read_csv(path2df)

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

    df.to_csv('df_clear.csv')
    return True


download_dataset()
clear_data('data.csv')
