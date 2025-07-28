import pandas as pd
import numpy as np
import os
import pickle


def main():
    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "raw_data")
    output_dir = os.path.join(parent_dir, "model_data")

    # Global capture production
    gcp_og = pd.read_csv(os.path.join(input_dir, "global_capture_production.csv"))

    y_start, y_end = 1950, 2021
    y_cols = list(range(y_start, y_end + 1))

    gcp_og = gcp_og.rename(
        columns={
            "ASFIS species (3-alpha code)": "Alpha3_Code",
            "FAO major fishing area (Code)": "Area",
            "Country (ISO3 code)": "ISO3",
            **{f"[{y}]": y for y in y_cols},
        }
    )

    # Remove rows with no reported landings in all years
    no_l_mask = (gcp_og[y_cols] == 0).all(axis=1)
    gcp_og = gcp_og[~no_l_mask]

    # FAO Effort data
    effort_og = pd.read_excel(
        os.path.join(input_dir, "FAOEffortBOB.xlsx"), sheet_name="MappedFAO"
    )

    effort_og = effort_og.rename(
        columns={
            "FAO": "Area",
            "EffortCellReportedEff": "Effort",
            "EffortCellIUUEff": "Effort_U",
        }
    )

    effort = (
        effort_og.groupby(["Area", "Year", "ISO3"])[["Effort", "Effort_U"]]
        .sum()
        .reset_index()
    )

    effort = effort.pivot(
        index=["Area", "ISO3"], columns="Year", values="Effort"
    ).reset_index()

    for y in set(y_cols) - set(effort.columns):
        effort[y] = np.nan

    eff_vals = effort[y_cols].values

    def interpolate_nan_rows(eff):
        """
        Replaces NaN values in a 2D NumPy array with linear interpolation/extrapolation
        across each row.

        Args:
            eff: A 2D NumPy array.

        Returns:
            A 2D NumPy array with NaN values interpolated/extrapolated row-wise.
        """
        eff_interpolated = np.copy(eff)
        n_rows = eff.shape[0]

        for i in range(n_rows):
            row = eff[i]
            nan_indices = np.where(np.isnan(row))[0]
            non_nan_indices = np.where(~np.isnan(row))[0]
            non_nan_values = row[non_nan_indices]

            if non_nan_indices.size > 0:
                interp_values = np.interp(nan_indices, non_nan_indices, non_nan_values)
                eff_interpolated[i, nan_indices] = interp_values

        return eff_interpolated

    eff_interpolated = interpolate_nan_rows(eff_vals)

    effort.loc[:, y_cols] = eff_interpolated

    cpue_og = pd.merge(
        gcp_og, effort, on=["Area", "ISO3"], suffixes=("_gcp", "_effort")
    )

    # Retrieve SOFIA data
    sofia = pd.read_excel(os.path.join(input_dir, "sofia2024.xlsx"))

    # Add rows for the combined species stocks
    mult_species = sofia[sofia["Alpha3_Code"].apply(lambda x: "," in x)][
        ["Area", "Alpha3_Code"]
    ].values

    gcp_cols = [f"{y}_gcp" for y in y_cols]
    eff_cols = [f"{y}_effort" for y in y_cols]

    for area, alphas in mult_species:
        alphas_list = alphas.split(", ")

        area_mask1 = cpue_og["Area"] == area
        alphas_mask1 = cpue_og["Alpha3_Code"].isin(alphas_list)

        temp = cpue_og[area_mask1 & alphas_mask1].copy()

        temp = temp.groupby("ISO3").agg(
            {**{col: "sum" for col in gcp_cols}, **{col: "first" for col in eff_cols}}
        )

        temp["Alpha3_Code"] = alphas
        temp["Area"] = area

        cpue_og = pd.concat([cpue_og, temp])

    pk = ["Area", "Alpha3_Code"]

    cpue = (
        cpue_og.groupby(["Area", "Alpha3_Code"])[gcp_cols + eff_cols]
        .sum()
        .reset_index()
    )

    for y in y_cols:
        cpue[y] = cpue[f"{y}_gcp"] / cpue[f"{y}_effort"] * 1e6

    sofia_cpue = pd.merge(sofia[pk], cpue[pk + y_cols], on=["Area", "Alpha3_Code"])

    with open(os.path.join(output_dir, "cpue.pkl"), "wb") as file:
        pickle.dump(sofia_cpue, file)


if __name__ == "__main__":
    main()
