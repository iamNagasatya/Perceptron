

def prepare_data(df, target_col="y"):
    x = df.drop(columns=target_col)
    y = df[target_col]

    return x, y

