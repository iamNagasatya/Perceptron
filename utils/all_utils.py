import logging

def prepare_data(df, target_col="y"):
    """It returns features and label for the given dataset

    Args:
        df (pd.DataFrame): This is a dataframe
        target_col (str, optional): label or targel column. Defaults to "y".

    Returns:
        tuple: features and label
    """
    logging.info("preparing data for training")
    x = df.drop(columns=target_col)
    y = df[target_col]

    return x, y

