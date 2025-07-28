import os
import pandas as pd
import copernicusmarine
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely import unary_union
import cartopy.crs as ccrs


def visualize_area_approximations(
    area_coords, file_path=os.path.dirname(os.getcwd()) + "/figures/"
):
    """Visualizes approximations of FAO major fishing areas.

    This function takes a dictionary of area coordinates and generates a plot
    showing the approximate boundaries of these areas.  It saves the plot as a PNG.

    Args:
        area_coords (dict): A dictionary where keys are area IDs and values are
            dictionaries containing "lat" and "long" keys, each holding slices
            defining the latitude and longitude ranges for the area.
        file_path (str, optional): The path to the directory where the figure
            will be saved. Defaults to "./figures/".

    Returns:
        None. The function saves the plot to a file.
    """
    plt.rcParams["font.family"] = "Georgia"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    label_adjustments = {
        21: (10, 0),
        27: (-5, 0),
        31: (5, 0),
        34: (-1, 0),
        37: (2, 5.2),
        41: (2, 0),
        47: (5, 0),
        51: (0, 0),
        57: (0, 0),
        61: (10, 0),
        67: (0, 0),
        71: [(5, 0), (-10, 0)],
        77: (0, 0),
        81: [(0, 0), (-5, 0)],
        87: (0, 0),
    }

    for area, coords in area_coords.items():
        polygons = []
        for lat_slice, lon_slice in zip(coords["lat"], coords["long"]):
            min_lon = lon_slice.start if lon_slice.start is not None else -180
            max_lon = lon_slice.stop if lon_slice.stop is not None else 180
            min_lat = lat_slice.start if lat_slice.start is not None else -90
            max_lat = lat_slice.stop if lat_slice.stop is not None else 90

            polygon = Polygon(
                [
                    (min_lon, min_lat),
                    (max_lon, min_lat),
                    (max_lon, max_lat),
                    (min_lon, max_lat),
                ]
            )
            polygons.append(polygon)

        combined_polygon = unary_union(polygons)

        if isinstance(combined_polygon, Polygon):
            label_x, label_y = label_adjustments[area]
            x, y = combined_polygon.exterior.xy
            ax.plot(x, y, transform=ccrs.PlateCarree(), color="blue", linewidth=1)
            ax.fill(x, y, transform=ccrs.PlateCarree(), color="gray", alpha=0.3)
            centroid = combined_polygon.centroid
            ax.text(
                centroid.x + label_x,
                centroid.y + label_y,
                str(area),
                transform=ccrs.PlateCarree(),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        elif isinstance(combined_polygon, MultiPolygon):
            for i, polygon in enumerate(combined_polygon.geoms):
                label_x, label_y = label_adjustments[area][i]
                x, y = polygon.exterior.xy
                ax.plot(x, y, transform=ccrs.PlateCarree(), color="blue", linewidth=1)
                ax.fill(x, y, transform=ccrs.PlateCarree(), color="gray", alpha=0.3)
                centroid = polygon.centroid
                ax.text(
                    centroid.x + label_x,
                    centroid.y + label_y,
                    str(area),
                    transform=ccrs.PlateCarree(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        else:
            print(f"Unexpected geometry type for area {area}: {type(combined_polygon)}")

    ax.coastlines(color="grey")
    ax.set_global()
    plt.title("FAO Major Fishing Areas Approximations")
    fig.savefig(os.path.join(file_path, "FAO_Area_Approximations.png"), pad_inches=0)


def get_variable_data(
    variable,
    dataset,
    area_coords,
    years,
    file_path,
    file_name="oceanography_dict.pkl",
    depth=0,
):
    """Builds and saves data for a specific variable from a dataset.

    This function processes a dataset, extracting data for a given variable,
    for specified areas and years. The processed data is then saved to a pickle file.

    Args:
        dataset (xarray.Dataset): The input dataset containing the variable.
        file_name (str): The name of the pickle file to save the data to.
        variable (str): The name of the variable to extract from the dataset.
        file_path (str, optional): The path to the directory where the pickle file
            will be saved. Defaults to "./model_data/".
        depth (int, optional): The depth level to select from the dataset, if applicable.
            Defaults to 0. If None, depth selection is skipped.

    Returns:
        None.  The function saves the data to a pickle file.
    """
    if os.path.exists(os.path.join(file_path, file_name)):
        with open(file_path + file_name, "rb") as file:
            data = pickle.load(file)
    else:
        data = {
            variable: {
                area: {year: pd.DataFrame() for year in years.keys()}
                for area in area_coords.keys()
            }
        }

    for area in tqdm(area_coords.keys(), desc=f"Areas for {variable}"):
        coords = area_coords[area]
        for year in tqdm(years.keys(), desc=f"Years for {variable} in area {area}"):
            year_slice = years[year]
            temp = pd.DataFrame()
            for lat, long in zip(coords["lat"], coords["long"]):
                if depth:
                    subset = (
                        dataset[[variable]]
                        .isel(depth=depth)
                        .sel(latitude=lat, longitude=long, time=year_slice)
                    )
                else:
                    subset = dataset[[variable]].sel(
                        latitude=lat, longitude=long, time=year_slice
                    )

                temp = pd.concat(
                    [
                        temp,
                        subset[variable].to_dataframe().dropna(),
                    ]
                )

            if variable not in data:
                data[variable] = {
                    area: {year: pd.DataFrame() for year in years.keys()}
                    for area in area_coords.keys()
                }
            data[variable][area][year] = (
                temp.groupby("time")[variable].mean().reset_index()
            )

    with open(os.path.join(file_path, file_name), "wb") as file:
        pickle.dump(data, file)

    print(f"{variable} data loaded onto {file_name} in {file_path}")


def main():
    # Create approximation of FAO major fishing areas
    # Areas are approximated as union of rectangular regions
    # in longitude, latitude space
    # Regions overlap with inland regions but this is fine
    # since collected data is only from ocean sources
    area_coords = {
        21: {"lat": [slice(35, 78)], "long": [slice(-80, -42)]},
        27: {
            "lat": [slice(36, 90), slice(43, 90), slice(49, 90)],
            "long": [slice(-42, -6), slice(-6, 0), slice(0, 68)],
        },
        31: {
            "lat": [slice(17, 35), slice(15, 17), slice(8.4, 15), slice(5, 8.4)],
            "long": [
                slice(-98, -40),
                slice(-90, -40),
                slice(-84, -40),
                slice(-63, -40),
            ],
        },
        34: {
            "lat": [slice(-6, 36), slice(-6, 8)],
            "long": [slice(-20, -5), slice(-5, 14)],
        },
        37: {
            "lat": [slice(34, 42), slice(30, 47)],
            "long": [slice(-5, 3), slice(3, 42)],
        },
        41: {
            "lat": [slice(-60, -34), slice(-50, 5)],
            "long": [slice(-67, -54), slice(-54, -20)],
        },
        47: {"lat": [slice(-40, -6)], "long": [slice(5, 30)]},
        51: {"lat": [slice(-45, 30)], "long": [slice(30, 77)]},
        57: {
            "lat": [slice(-55, 24), slice(-55, -1), slice(-55, -8), slice(-55, -30)],
            "long": [slice(77, 100), slice(100, 105), slice(105, 129), slice(129, 150)],
        },
        61: {
            "lat": [slice(15, 23), slice(20, 65)],
            "long": [slice(105, 115), slice(115, 180)],
        },
        67: {"lat": [slice(40, 66)], "long": [slice(-175, -123)]},
        71: {
            "lat": [slice(-8, 15), slice(15, 20), slice(-25, -8)],
            "long": [slice(99, 129), slice(115, 175), slice(129, 175)],
        },
        77: {
            "lat": [slice(-25, 40), slice(5, 40), slice(5, 15), slice(5, 8)],
            "long": [
                slice(-175, -120),
                slice(-120, -98),
                slice(-98, -84),
                slice(-84, -78),
            ],
        },
        81: {
            "lat": [slice(-60, -25), slice(-60, -25)],
            "long": [slice(150, 180), slice(-180, -120)],
        },
        87: {
            "lat": [slice(-60, -55), slice(-55, 5)],
            "long": [slice(-120, -67), slice(-120, -69)],
        },
    }

    # Save visualization of approximated FAO areas
    visualize_area_approximations(area_coords)

    # Define years for data
    years = {
        year: slice(f"{year}-01-01", f"{year}-12-31") for year in range(1950, 2022)
    }

    # First Copernicus Marine dataset to be used in oceanographic data
    # Found in CM Data Store under Global Ocean Ensemble Physics Reanalysis
    # doi.org/10.48670/moi-00024
    dataset1 = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-mnstd_my_0.25deg_P1M-m",
        username=os.getenv("copernicus_username"),
        password=os.getenv("copernicus_password"),
    )

    # Found in CM Data Store under Global Ocean Biogeochemistry Hindcast
    # doi.org/10.48670/moi-00019
    dataset2 = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1M-m",
        username=os.getenv("copernicus_username"),
        password=os.getenv("copernicus_password"),
    )

    # Maps variable names to which dataset they belong to
    build_data_dict = {
        "thetao_mean": {"dataset": dataset1, "depth": 0},
        "so_mean": {"dataset": dataset1, "depth": 0},
        "chl": {"dataset": dataset2, "depth": 0},
        "no3": {"dataset": dataset2, "depth": 0},
        "po4": {"dataset": dataset2, "depth": 0},
        "si": {"dataset": dataset2, "depth": 0},
        "nppv": {"dataset": dataset2, "depth": 0},
        "o2": {"dataset": dataset2, "depth": 0},
        "fe": {"dataset": dataset2, "depth": 0},
        "phyc": {"dataset": dataset2, "depth": 0},
        "ph": {"dataset": dataset2, "depth": 0},
        "spco2": {"dataset": dataset2, "depth": None},
    }

    parent_dir = os.getcwd()
    output_dir = os.path.join(parent_dir, "model_data")

    for var, var_info in build_data_dict.items():
        dataset = var_info["dataset"]
        depth = var_info["depth"]

        get_variable_data(var, dataset, area_coords, years, output_dir, depth=depth)

    # Reformat dictionary into Dataframe with all variable info
    with open(os.path.join(output_dir, "oceanography_dict.pkl"), "rb") as file:
        o_dict = pickle.load(file)

    o_df = pd.DataFrame()

    for var, var_info in o_dict.items():
        for area, area_info in var_info.items():
            for year, df in area_info.items():
                if not df.empty:
                    temp = df.copy()
                    temp = temp.rename(columns={var: "Value"})
                    temp["Var"] = var
                    temp["Area"] = area

                    o_df = pd.concat([o_df, temp])

    with open(os.path.join(output_dir, "oceanography.pkl"), "wb") as file:
        pickle.dump(o_df, file)


if __name__ == "__main__":
    main()
