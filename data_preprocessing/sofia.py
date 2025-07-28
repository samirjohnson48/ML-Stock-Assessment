import pandas as pd
import numpy as np
import os
import pickle


def create_target_values(X):
    """Creates target values based on input stock assessed status.

    This function takes a string representing fishing activity and returns a
    numerical target value based on the presence and combination of "O" (Overfished),
    "F" (Fully Fished), and "U" (Underfished) indicators.

    Args:
        X (str): Stock assessment(s) of stock(s) for a given species,
        containing separators such as "-", "/", ",", or ";".

    Returns:
        int: A target value based on the following rules:
            - 1: Status is consistently "F" or "U" across stocks
            - 0: Status is consistently "O" across stocks
            - -1: Status is absent or inconsistent across stocks
    """
    X = X.strip()
    if len(X) == 1:
        if X not in ["O", "F", "U"]:
            return -1
        return 1 - int(X == "O")

    if "-" in X or "/" in X:
        sep = "-" if "-" in X else "/"

        s = X.split(sep)
        s = [st.strip() for st in s]

        for st in s:
            if st in ["O", "F", "U"]:
                return 1 - int(st == "O")
        return -1

    elif "," in X or ";" in X:
        sep = "," if "," in X else ";"

        s = X.split(sep)
        s = [st.strip() for st in s]

        if ("O" in s and "F" in s) or ("O" in s and "U" in s):
            return -1
        elif "O" in s:
            return 0
        elif "F" in s or "U" in s:
            return 1
        return -1

    return -1


def main():
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "raw_data")
    output_dir = os.path.join(parent_dir, "model_data")

    # Read in raw sofia data
    sofia_og = pd.read_excel(os.path.join(input_dir, "sofia2024.xlsx"))

    # Convert sofia data from wide to long format
    sofia = pd.wide_to_long(
        sofia_og.reset_index(), ["X", "U"], i=["Area", "Alpha3_Code", "index"], j="Year"
    ).reset_index()[["Area", "Alpha3_Code", "Year", "X", "U"]]

    # Drop rows with no status
    sofia = sofia.dropna(subset="X")

    # Create binary target values for status
    # Essentially maps U, F --> 1 and O --> 0
    # Aggregates status across stocks to create label for species
    # If status is inconsistent across stocks for given species in area,
    # this observation is dropped
    sofia["S"] = sofia["X"].apply(create_target_values)
    sofia = sofia[sofia["S"] != -1].reset_index(drop=True)

    with open(os.path.join(output_dir, "target.pkl"), "wb") as file:
        pickle.dump(sofia, file)


if __name__ == "__main__":
    main()
