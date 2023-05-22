import logging

def prepare_data(df, target_col="y"):
    logging.info("preparing data for training")
    x = df.drop(columns=target_col)
    y = df[target_col]

    return x, y

