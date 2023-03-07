"""
A collection of utility functions.
"""


def log_features(df, log="features") -> None:
    """
    Outputs a list of existing features and their datatypes to a log file.

    Args:
        df: The dataframe representing the TARCC dataset.
        log: The name of the log file.

    Returns:
        None
    """
    with open(f"{log}.log", "w") as f:
        for i in range(len(df.columns)):
            f.write(f"{df.columns[i]}\t{df.dtypes[i]}\n")
