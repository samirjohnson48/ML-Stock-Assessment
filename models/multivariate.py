import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from math import floor
from itertools import product, combinations
import gc
import argparse
import logging  # Added for improved logging

from sktime.transformations.panel.rocket import MultiRocket
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.decomposition import (
    PCA,
)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def format_data(
    cpue_df: pd.DataFrame,
    target_df: pd.DataFrame,
    oceanography_df: pd.DataFrame,
    vars_to_include: list,
) -> tuple:
    """
    Formats and splits data for model training and testing, combining CPUE and oceanographic time series.

    This function merges CPUE timeseries data with target data, extracts and aligns
    CPUE and selected oceanographic variable time series, truncates them based on
    the target year, pads them to a consistent length, and splits the data.

    Args:
        cpue_df (pd.DataFrame): DataFrame containing CPUE timeseries data with
            "Area", "Species", "Year", and 'CPUE' columns (annual data).
        target_df (pd.DataFrame): DataFrame containing target data with "Area",
            "Alpha3_Code", and "S" (target variable) columns.
        oceanography_df (pd.DataFrame): DataFrame with 'time' (datetime), 'Value',
            'Var' (string, e.g., 'SST', 'Salinity'), and 'Area' (int) (monthly data).
        vars_to_include (list): List of strings specifying which 'Var' values from
                                 oceanography_df to include as channels.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test, and X_test_metadata, which are
            the training and testing data for the model. X_train and X_test are
            NumPy arrays of shape (samples, n_channels, timeseries_length) and type float32.
            y_train and y_test are NumPy arrays of shape (samples,) containing the
            target variable.
    """
    logging.info("Starting multivariate data formatting...")

    # Use clearer variable names and ensure robust merging
    model_data = pd.merge(cpue_df, target_df, on=["Area", "Alpha3_Code"], how="inner")
    logging.info(f"Merged CPUE and target data contains {len(model_data)} records.")

    # Determine dynamic year range from CPUE columns for filtering
    cpue_years_cols = [
        col
        for col in cpue_df.columns
        if isinstance(col, int)
        and col >= 1950
        and col <= 2023  # Keep bounds for sanity
    ]
    if not cpue_years_cols:
        raise ValueError(
            "No valid year columns found in cpue_df for time series extraction."
        )

    min_cpue_year = min(cpue_years_cols)
    max_cpue_year = max(cpue_years_cols)
    logging.info(
        f"Dynamic CPUE data range detected: {min_cpue_year} to {max_cpue_year}."
    )

    # Determine overall fixed start date for monthly time series (based on oceanography start)
    # If oceanography_df is empty or has no time data, default to a safe start
    if oceanography_df.empty or "time" not in oceanography_df.columns:
        raise ValueError("Oceanography DataFrame is empty or missing 'time' column.")

    min_ocean_date = oceanography_df["time"].min()
    start_date_fixed = min_ocean_date.to_period(
        "M"
    ).start_time  # Ensure it's month start

    logging.info(
        f"Fixed monthly time series start date: {start_date_fixed.strftime('%Y-%m-%d')}"
    )

    all_ts_channels = []  # To store all [channel1_ts, channel2_ts, ...] for each sample
    valid_sample_indices = []

    for idx, row in model_data.iterrows():
        target_year = int(row["Year"])
        area_id = row["Area"]
        alpha3_code = row["Alpha3_Code"]

        # Define end date for current sample (end of target year)
        end_date_for_sample = (
            pd.Timestamp(target_year, 12, 31).to_period("M").end_time
        )  # End of the target year's last month

        # Generate full monthly date range for this sample
        monthly_dates_for_sample = pd.date_range(
            start=start_date_fixed, end=end_date_for_sample, freq="MS"
        )

        if monthly_dates_for_sample.empty:
            logging.debug(
                f"Skipping sample {idx} (Area: {area_id}, Species: {alpha3_code}, Year: {target_year}) due to empty monthly date range."
            )
            continue

        sample_channels = []

        # --- Process CPUE Channel ---
        cpue_ts_annual_vals = []
        for y in range(min_cpue_year, target_year + 1):
            if y in cpue_years_cols and y in row.index:
                cpue_ts_annual_vals.append((y, row[y]))

        # Create a yearly Series for easier reindexing
        cpue_series_annual = pd.Series(
            [val for _, val in cpue_ts_annual_vals],
            index=[pd.Period(year, freq="Y") for year, _ in cpue_ts_annual_vals],
        )

        # Convert annual CPUE to monthly by repeating the annual value for each month
        # Use a temporary monthly index spanning the annual range to perform ffill
        temp_monthly_index = pd.date_range(
            start=pd.Timestamp(min_cpue_year, 1, 1),
            end=pd.Timestamp(target_year, 12, 31),
            freq="MS",
        )

        cpue_monthly_series = pd.Series(index=temp_monthly_index, dtype=np.float32)
        for year, value in cpue_ts_annual_vals:
            year_start = pd.Timestamp(year, 1, 1)
            year_end = pd.Timestamp(year, 12, 31)
            cpue_monthly_series.loc[year_start:year_end] = np.float32(value)

        # Now reindex to the specific monthly_dates_for_sample (from 1993-01 to target_year-12)
        # Any months before min_cpue_year (e.g., 1993-1950) will be NaNs
        cpue_channel = (
            cpue_monthly_series.reindex(monthly_dates_for_sample)
            .fillna(0)
            .values.astype(np.float32)
        )  # Fill leading NaNs with 0

        sample_channels.append(cpue_channel)

        # --- Process Oceanographic Channels ---
        for var_name in vars_to_include:
            ocean_var_df = oceanography_df[
                (oceanography_df["Area"] == area_id)
                & (oceanography_df["Var"] == var_name)
            ].set_index("time")["Value"]

            # Reindex to the desired monthly range and fill missing values
            ocean_channel = (
                ocean_var_df.reindex(monthly_dates_for_sample, method="nearest")
                .fillna(0)
                .values.astype(np.float32)
            )
            # Using 'nearest' then 'fillna(0)' might be better than ffill/bfill for initial missing data.

            sample_channels.append(ocean_channel)

        # Check if all channels for this sample have the same length
        channel_lengths = [len(c) for c in sample_channels]
        if not all(l == channel_lengths[0] for l in channel_lengths):
            logging.error(
                f"Channel lengths mismatch for sample (Area: {area_id}, Species: {alpha3_code}, Year: {target_year}). Skipping."
            )
            continue  # Skip this sample if channels are not aligned

        all_ts_channels.append(
            np.stack(sample_channels)
        )  # Stack channels for this sample
        valid_sample_indices.append(idx)

    model_data_filtered = model_data.loc[valid_sample_indices].copy()

    if not all_ts_channels:
        raise ValueError(
            "No valid time series data found after formatting and channel creation. "
            "Check 'cpue_df', 'target_df', 'oceanography_df' contents, year range logic, and 'vars_to_include'."
        )

    # Determine max time series length across all samples and channels
    max_ts_length = max(ts.shape[1] for ts in all_ts_channels)
    logging.info(f"Maximum time series length (in months) found: {max_ts_length}.")

    # Pad all channels to the maximum length
    X_padded = []
    for ts_channels in all_ts_channels:
        num_channels = ts_channels.shape[0]
        current_len = ts_channels.shape[1]

        if current_len < max_ts_length:
            padded_channels = []
            for channel in ts_channels:
                padded_channel = np.pad(
                    channel,
                    (0, max_ts_length - current_len),
                    "constant",
                    constant_values=0,
                )
                padded_channels.append(padded_channel)
            X_padded.append(np.stack(padded_channels))
        else:
            X_padded.append(ts_channels)

    X = np.array(X_padded).astype(np.float32)
    y = model_data_filtered["S"].values.astype(np.float32)

    logging.info(f"Splitting data into training (80%) and testing (20%) sets.")
    X_train, X_test, y_train, y_test, _, X_test_metadata = train_test_split(
        X,
        y,
        model_data_filtered[["Area", "Alpha3_Code", "Year"]],
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logging.info("Multivariate data formatting complete.")

    return X_train, X_test, y_train, y_test, X_test_metadata


def create_model_pipeline(
    encoder_args: dict,
    clf_args: dict,
    pca_args: dict = None,
    scaler_args: dict = None,
    sampler_args: dict = None,
    classifier_type: str = "logistic regression",
) -> make_pipeline:
    """
    Creates a scikit-learn pipeline for the model.
    The steps of the pipeline just contain the objects, and make_pipeline
    will automatically assign default string names.

    Args:
        encoder_args (dict): Arguments for the MultiRocket encoder.
        clf_args (dict): Arguments for the classifier head.
        pca_args (dict, optional): Arguments for PCA. If None, PCA is skipped. Defaults to None.
        scaler_args (dict, optional): Arguments for StandardScaler. If None, StandardScaler is skipped. Defaults to None.
        sampler_args (dict, optional): Arguments for ADASYN sampler. If None, ADASYN is skipped. Defaults to None.
        classifier_type (str, optional): Type of classifier ('logistic regression' or 'random forest').
                                        Defaults to "logistic regression".

    Returns:
        make_pipeline: A scikit-learn pipeline object.
    """
    pipeline_steps = []

    # Add MultiRocket encoder
    pipeline_steps.append(MultiRocket(**encoder_args))
    logging.debug(f"Added MultiRocket with args: {encoder_args}")

    # Add PCA if pca_args are provided (i.e., not None)
    if pca_args:
        pipeline_steps.append(PCA(**pca_args))
        logging.debug(f"Added PCA with args: {pca_args}")

    # Add StandardScaler if scaler_args are provided (though typically always used)
    if scaler_args:
        pipeline_steps.append(StandardScaler(**scaler_args))
        logging.debug(f"Added StandardScaler with args: {scaler_args}")

    # Add ADASYN sampler if sampler_args are provided
    if sampler_args:
        pipeline_steps.append(ADASYN(**sampler_args))
        logging.debug(f"Added ADASYN with args: {sampler_args}")

    # Add the classifier head
    if classifier_type.lower() == "logistic regression":
        pipeline_steps.append(LogisticRegression(**clf_args))
        logging.debug(f"Added LogisticRegression with args: {clf_args}")
    elif classifier_type.lower() == "random forest":
        pipeline_steps.append(RandomForestClassifier(**clf_args))
        logging.debug(f"Added RandomForestClassifier with args: {clf_args}")
    else:
        raise ValueError(
            f"Unsupported classifier type: {classifier_type}. Choose 'logistic regression' or 'random forest'."
        )

    pipeline = make_pipeline(*pipeline_steps)
    logging.info("Model pipeline created successfully.")
    return pipeline


def run_experiment(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, config: dict, output_base_dir: str, vars_used_in_format: list = None):
    """
    Runs a single model training and evaluation experiment for a given configuration.
    Includes cross-validation metrics, and evaluates the final model on a separate holdout test set.

    Args:
        X_train (np.ndarray): The training feature set for cross-validation and final model training.
        X_test (np.ndarray): The holdout test feature set for final, unbiased evaluation.
        y_train (np.ndarray): The training target set for cross-validation and final model training.
        y_test (np.ndarray): The holdout test target set for final, unbiased evaluation.
        config (dict): Dictionary containing MultiRocket parameters (num_kernels,
                       max_dilations, n_features) and potentially C.
        output_base_dir (str): The base directory where all experiment outputs will be stored.
        vars_used_in_format (list, optional): List of oceanographic variables used to format X_train/X_test.
                                              Used for naming output directories.

    Returns:
        dict: A dictionary of results for the current configuration, or None if an error occurred.
    """
    nk = config.get("num_kernels")
    mdpk = config.get("max_dilations")
    nfpk = config.get("n_features")
    C_val = config.get("C", 0.01) # Default C if not specified

    # Construct a unique label for this configuration
    if vars_used_in_format is not None:
        vars_label = "_".join(sorted(vars_used_in_format)) if vars_used_in_format else "CPUE_Only"
        config_label = f"VARS={vars_label}_NK={nk}_MDPK={mdpk}_NFPK={nfpk}_C={C_val}"
    else:
        config_label = f"NK={nk}_MDPK={mdpk}_NFPK={nfpk}_C={C_val}"

    model_specific_output_dir = os.path.join(output_base_dir, config_label)
    os.makedirs(model_specific_output_dir, exist_ok=True)
    
    logging.info(f"--- Starting experiment for configuration: {config_label} ---")
    
    current_encoder_args = {
        "num_kernels": nk,
        "max_dilations_per_kernel": mdpk,
        "n_features_per_kernel": nfpk,
        "normalise": True,
        "n_jobs": 1, 
        "random_state": 123,
    }
    
    fixed_lr_clf_args = {
        "C": C_val,
        "random_state": 123, "solver": "liblinear", 
        "penalty": "l2", "max_iter": 100, "class_weight": "balanced",
    }
    scaler_args = {"with_mean": True, "with_std": True}
    sampler_args_for_use = {"sampling_strategy": 0.5, "random_state": 123}
    
    results_data = {
        'nk': nk,
        'mdpk': mdpk,
        'nfpk': nfpk,
        'C': C_val,
        'vars_included': ", ".join(sorted(vars_used_in_format)) if vars_used_in_format else "CPUE_Only",
        'mean_cv_accuracy': np.nan, 
        'std_cv_accuracy': np.nan,   
        'mean_cv_balanced_accuracy': np.nan, 
        'std_cv_balanced_accuracy': np.nan,   
        'holdout_test_accuracy': np.nan, 
        'holdout_test_balanced_accuracy': np.nan, 
        'training_time_seconds': np.nan,
        'confusion_matrix_path': None,
        'model_path': None,
        'status': 'Failed'
    }

    try:
        current_pipeline = create_model_pipeline(
            encoder_args=current_encoder_args,
            clf_args=fixed_lr_clf_args,
            pca_args=None,
            scaler_args=scaler_args,
            sampler_args=sampler_args_for_use,
            classifier_type="logistic regression",
        )

        logging.info("Performing K-Fold Cross-Validation on training data...")
        start_time = time()
        
        cv_folds = KFold(n_splits=10, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score)
        }

        cv_results = cross_validate(
            current_pipeline,
            X_train, y_train, 
            cv=cv_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
            verbose=3
        )
        
        tot_time = time() - start_time
        mins = floor(tot_time / 60)
        secs = tot_time % 60
        logging.info(f"Cross-Validation complete. Total CV time: {mins} minutes, {secs:.2f} seconds")
        results_data['training_time_seconds'] = tot_time 

        # Extract and store mean and std of CV scores
        mean_acc_cv = np.mean(cv_results['test_accuracy'])
        std_acc_cv = np.std(cv_results['test_accuracy'])
        mean_b_acc_cv = np.mean(cv_results['test_balanced_accuracy'])
        std_b_acc_cv = np.std(cv_results['test_balanced_accuracy'])

        logging.info(f"  Mean CV Accuracy: {mean_acc_cv:.4f} (Std: {std_acc_cv:.4f})")
        logging.info(f"  Mean CV Balanced Accuracy: {mean_b_acc_cv:.4f} (Std: {std_b_acc_cv:.4f})\n")
        results_data['mean_cv_accuracy'] = mean_acc_cv
        results_data['std_cv_accuracy'] = std_acc_cv
        results_data['mean_cv_balanced_accuracy'] = mean_b_acc_cv
        results_data['std_cv_balanced_accuracy'] = std_b_acc_cv
        
        # Train a final model on the entire training dataset (X_train, y_train) for saving and holdout evaluation
        logging.info("Training final model on full training dataset for saving and holdout evaluation...")
        final_model_fit_start_time = time()
        current_pipeline.fit(X_train, y_train) 
        final_model_fit_time = time() - final_model_fit_start_time
        logging.info(f"Final model training time: {final_model_fit_time:.2f} seconds")

        # Evaluate the final trained model on the separate holdout test set (X_test, y_test)
        logging.info("Evaluating final model on separate holdout test set...")
        y_pred_holdout = current_pipeline.predict(X_test)

        holdout_acc = accuracy_score(y_test, y_pred_holdout)
        holdout_b_acc = balanced_accuracy_score(y_test, y_pred_holdout)
        
        logging.info(f"  Holdout Test Accuracy: {holdout_acc:.4f}")
        logging.info(f"  Holdout Test Balanced Accuracy: {holdout_b_acc:.4f}\n")
        results_data['holdout_test_accuracy'] = holdout_acc
        results_data['holdout_test_balanced_accuracy'] = holdout_b_acc

        # --- Plot Confusion Matrix for the final model on the holdout test data ---
        cm = confusion_matrix(y_test, y_pred_holdout)
        logging.info(f"Confusion Matrix (from final model on holdout test data):\n%s", cm)

        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {config_label}\n(Evaluated on Holdout Test Set)", fontsize=14)

        plot_path = os.path.join(model_specific_output_dir, "confusion_matrix_holdout_test_data.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Confusion Matrix plot saved to {plot_path}")
        results_data['confusion_matrix_path'] = plot_path

        # Save the final trained model
        model_path = os.path.join(model_specific_output_dir, "final_trained_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(current_pipeline, f)
        logging.info(f"Final trained model saved to {model_path}")
        results_data['model_path'] = model_path
        results_data['status'] = 'Success'

    except Exception as e:
        logging.error(f"An unexpected error occurred for config {config_label}: {e}", exc_info=True)
        # Results data already initialized with np.nan and 'Failed' status

    logging.info(f"--- Experiment for {config_label} Complete ---")
    return results_data


def run_c_grid_search(
    X_train,
    X_test,
    y_train,
    y_test,
    rocket_config: dict,
    output_base_dir: str,
    C_values: list,
):
    """
    Performs a GridSearchCV over the C parameter for Logistic Regression
    for a given MultiRocket configuration.

    Args:
        X_train, X_test, y_train, y_test: Data splits.
        rocket_config (dict): Fixed MultiRocket parameters (num_kernels, max_dilations, n_features).
        output_base_dir (str): Base directory for outputs.
        C_values (list): List of C values to search over.

    Returns:
        list: A list of dictionaries, where each dictionary contains results for a specific C value.
    """
    nk = rocket_config.get("num_kernels")
    mdpk = rocket_config.get("max_dilations")
    nfpk = rocket_config.get("n_features")

    config_label = f"NK={nk}_MDPK={mdpk}_NFPK={nfpk}"
    model_specific_output_dir = os.path.join(output_base_dir, config_label)
    os.makedirs(model_specific_output_dir, exist_ok=True)

    logging.info(
        f"--- Starting GridSearchCV for C on configuration: {config_label} ---"
    )

    encoder_args = {
        "num_kernels": nk,
        "max_dilations_per_kernel": mdpk,
        "n_features_per_kernel": nfpk,
        "normalise": True,
        "n_jobs": 1,
        "random_state": 123,
    }
    scaler_args = {"with_mean": True, "with_std": True}
    sampler_args_for_use = {"sampling_strategy": 0.5, "random_state": 123}

    # Create the base pipeline steps without the final classifier
    # The classifier will be replaced by GridSearchCV's estimator
    base_pipeline_steps = []
    base_pipeline_steps.append(MultiRocket(**encoder_args))
    if scaler_args:
        base_pipeline_steps.append(StandardScaler(**scaler_args))
    if sampler_args_for_use:
        base_pipeline_steps.append(ADASYN(**sampler_args_for_use))

    # The estimator for GridSearchCV is the LogisticRegression
    grid_estimator = make_pipeline(
        *base_pipeline_steps,
        LogisticRegression(
            random_state=123,
            solver="liblinear",
            penalty="l2",
            max_iter=100,
            class_weight="balanced",
        ),
    )

    param_grid = {"logisticregression__C": C_values}

    # Define scorers for GridSearchCV
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    grid_search = GridSearchCV(
        estimator=grid_estimator,
        param_grid=param_grid,
        cv=KFold(
            n_splits=5, shuffle=True, random_state=42
        ),  # Using KFold for robust CV
        scoring=scoring,
        refit="balanced_accuracy",  # Refit the best estimator based on balanced accuracy
        n_jobs=-1,  # Use all available cores for grid search
        verbose=3,
        return_train_score=True,
    )

    start_time = time()
    grid_search.fit(X_train, y_train)
    tot_time = time() - start_time
    mins = floor(tot_time / 60)
    secs = tot_time % 60
    logging.info(f"GridSearchCV total time: {mins} minutes, {secs:.2f} seconds")

    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best balanced accuracy (CV score): {grid_search.best_score_:.4f}")

    # Evaluate the best estimator on the test set
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_b_acc = balanced_accuracy_score(y_test, y_pred)
    logging.info(f"Best estimator test set Accuracy: {test_acc:.4f}")
    logging.info(f"Best estimator test set Balanced Accuracy: {test_b_acc:.4f}")

    # Save the best model
    best_model_path = os.path.join(
        model_specific_output_dir, f"best_model_C_search.pkl"
    )
    with open(best_model_path, "wb") as f:
        pickle.dump(best_estimator, f)
    logging.info(f"Best model from C search saved to {best_model_path}")

    # Collect results for plotting (for each C value)
    cv_results = grid_search.cv_results_
    c_search_results = []
    for i, C_val in enumerate(cv_results["param_logisticregression__C"]):
        res_dict = {
            "nk": nk,
            "mdpk": mdpk,
            "nfpk": nfpk,
            "C": C_val,
            "mean_cv_accuracy": cv_results[f"mean_test_accuracy"][i],
            "std_cv_accuracy": cv_results[f"std_test_accuracy"][i],
            "mean_cv_balanced_accuracy": cv_results[f"mean_test_balanced_accuracy"][i],
            "std_cv_balanced_accuracy": cv_results[f"std_test_balanced_accuracy"][i],
            "test_accuracy_best_model": (
                test_acc
                if C_val == grid_search.best_params_["logisticregression__C"]
                else np.nan
            ),
            "test_balanced_accuracy_best_model": (
                test_b_acc
                if C_val == grid_search.best_params_["logisticregression__C"]
                else np.nan
            ),
            "training_time_seconds": tot_time,  # Total time for grid search
            "model_path": (
                best_model_path
                if C_val == grid_search.best_params_["logisticregression__C"]
                else None
            ),
            "status": "Success",  # Assume success if grid search completed
        }
        c_search_results.append(res_dict)

    logging.info(
        f"--- GridSearchCV for C on configuration: {config_label} Complete ---"
    )
    return c_search_results


def run_vars_grid_search(
    cpue_df: pd.DataFrame,
    target_df: pd.DataFrame,
    oceanography_df: pd.DataFrame,
    all_possible_ocean_vars: list,
    rocket_config: dict,
    fixed_lr_clf_C: float,
    output_base_dir: str,
) -> list:
    """
    Performs a grid search over all combinations of oceanographic input variables.

    Args:
        cpue_df, target_df, oceanography_df: Raw dataframes.
        all_possible_ocean_vars (list): List of all possible oceanographic variables to consider.
        rocket_config (dict): Fixed MultiRocket parameters.
        fixed_lr_clf_C (float): Fixed C value for Logistic Regression.
        output_base_dir (str): Base directory for outputs.

    Returns:
        list: A list of dictionaries, where each dictionary contains results for a specific variable combination.
    """
    logging.info(
        "--- Starting Grid Search over Oceanographic Variable Combinations ---"
    )

    all_results_vars_search = []

    # Generate all non-empty combinations of variables
    # This includes combinations of size 1 up to the full set
    all_combinations = []
    for i in range(len(all_possible_ocean_vars) + 1):
        all_combinations.extend(combinations(all_possible_ocean_vars, i))

    # Filter out the empty combination if you always want at least CPUE + some ocean var
    # If you want CPUE-only as a baseline, keep the empty tuple and handle it.
    # For this implementation, we'll include CPUE-only (empty tuple for ocean vars)

    # Sort combinations by length and then alphabetically for consistent logging/plotting
    all_combinations.sort(key=lambda x: (len(x), tuple(sorted(x))))

    # Add CPUE-only baseline if not implicitly covered
    if () in all_combinations:
        logging.info("Including 'CPUE_Only' as a baseline combination.")
    else:
        all_combinations.insert(0, ())  # Ensure CPUE-only is first if not already there

    for i, vars_subset_tuple in enumerate(all_combinations):
        vars_subset = list(vars_subset_tuple)  # Convert tuple to list for format_data

        logging.info(
            f"Running experiment for variable combination {i+1}/{len(all_combinations)}: {vars_subset if vars_subset else 'CPUE_Only'}"
        )

        try:
            # Re-format data for the current subset of variables
            (
                X_train_subset,
                X_test_subset,
                y_train_subset,
                y_test_subset,
                X_test_metadata_subset,
            ) = format_data(cpue_df, target_df, oceanography_df, vars_subset)

            # Prepare config for run_experiment
            current_config = {
                **rocket_config,  # Unpack fixed MultiRocket params
                "C": fixed_lr_clf_C,  # Add fixed C
            }

            # Run the experiment for this variable combination
            result = run_experiment(
                X_train_subset,
                X_test_subset,
                y_train_subset,
                y_test_subset,
                current_config,
                output_base_dir,
                vars_used_in_format=vars_subset,
            )
            if result:
                all_results_vars_search.append(result)
        except Exception as e:
            vars_label = ", ".join(sorted(vars_subset)) if vars_subset else "CPUE_Only"
            logging.error(
                f"Error during variable combination search for {vars_label}: {e}",
                exc_info=True,
            )
            # Log a failed entry for this combination
            failed_result = {
                "nk": rocket_config.get("num_kernels"),
                "mdpk": rocket_config.get("max_dilations"),
                "nfpk": rocket_config.get("n_features"),
                "C": fixed_lr_clf_C,
                "vars_included": vars_label,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "training_time_seconds": np.nan,
                "confusion_matrix_path": None,
                "model_path": None,
                "status": "Failed",
            }
            all_results_vars_search.append(failed_result)

        gc.collect()  # Clean up memory after each run

    logging.info("--- Oceanographic Variable Combinations Grid Search Complete ---")
    return all_results_vars_search


def plot_vars_search_results(
    results_df: pd.DataFrame,
    rocket_config: dict,
    fixed_lr_clf_C: float,
    output_base_dir: str,
):
    """
    Plots the accuracy and balanced accuracy scores for each oceanographic variable combination.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the variable combination search.
                                   Expected to have 'vars_included', 'accuracy', 'balanced_accuracy' columns.
        rocket_config (dict): Fixed MultiRocket parameters used for this search.
        fixed_lr_clf_C (float): Fixed C value used for this search.
        output_base_dir (str): Base directory for outputs.
    """
    if results_df.empty:
        logging.warning("No variable search results to plot performance metrics.")
        return

    logging.info("Generating oceanographic variable combination performance plot...")

    # Ensure only successful runs are plotted
    plot_df = results_df[results_df["status"] == "Success"].copy()
    if plot_df.empty:
        logging.warning("No successful variable combination runs to plot.")
        return

    # Sort by balanced accuracy for better visualization
    plot_df = plot_df.sort_values(by="balanced_accuracy", ascending=False).reset_index(
        drop=True
    )

    # Create a label for each combination
    plot_df["combination_label"] = plot_df["vars_included"].apply(
        lambda x: "CPUE_Only" if x == "CPUE_Only" else "CPUE + " + x
    )

    nk = rocket_config.get("num_kernels")
    mdpk = rocket_config.get("max_dilations")
    nfpk = rocket_config.get("n_features")

    # Create a specific directory for this plot within the overall output structure
    plot_output_dir = os.path.join(
        output_base_dir,
        f"Variable_Search_NK={nk}_MDPK={mdpk}_NFPK={nfpk}_C={fixed_lr_clf_C}",
    )
    os.makedirs(plot_output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(
        figsize=(16, 9)
    )  # Larger figure for potentially many combinations

    # Plot Accuracy
    ax1.bar(
        plot_df["combination_label"],
        plot_df["accuracy"],
        width=0.4,
        label="Accuracy",
        align="center",
        color="skyblue",
    )
    ax1.set_ylabel("Accuracy", color="skyblue")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    ax1.set_ylim(0, 1)

    # Create a second y-axis for Balanced Accuracy
    ax2 = ax1.twinx()
    ax2.bar(
        plot_df["combination_label"],
        plot_df["balanced_accuracy"],
        width=0.4,
        label="Balanced Accuracy",
        align="edge",
        color="lightcoral",
        alpha=0.7,
    )
    ax2.set_ylabel("Balanced Accuracy", color="lightcoral")
    ax2.tick_params(axis="y", labelcolor="lightcoral")
    ax2.set_ylim(0, 1)

    ax1.set_xlabel("Oceanographic Variable Combination")
    plt.title(
        f"Performance for Different Oceanographic Variable Combinations\n(MultiRocket: NK={nk}, MDPK={mdpk}, NFPK={nfpk}; LR C={fixed_lr_clf_C})",
        fontsize=16,
    )

    # Rotate x-axis labels for readability
    ax1.set_xticklabels(
        plot_df["combination_label"], rotation=70, ha="right", fontsize=10
    )

    fig.tight_layout()

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    plot_path = os.path.join(plot_output_dir, "ocean_vars_combination_performance.png")
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(
        f"Oceanographic variable combination performance plot saved to {plot_path}"
    )


def plot_c_search_results(
    c_search_results: list, rocket_config: dict, output_base_dir: str
):
    """
    Plots the accuracy and balanced accuracy scores vs. C values from a GridSearchCV.

    Args:
        c_search_results (list): List of dictionaries, each containing results for a C value.
        rocket_config (dict): Fixed MultiRocket parameters (num_kernels, max_dilations, n_features).
        output_base_dir (str): Base directory for outputs.
    """
    if not c_search_results:
        logging.warning("No C search results to plot.")
        return

    logging.info("Generating C parameter search results plot...")

    df_results = pd.DataFrame(c_search_results)

    # Ensure C values are sorted for plotting
    df_results = df_results.sort_values(by="C").reset_index(drop=True)

    nk = rocket_config.get("num_kernels")
    mdpk = rocket_config.get("max_dilations")
    nfpk = rocket_config.get("n_features")
    config_label = f"NK={nk}_MDPK={mdpk}_NFPK={nfpk}"
    model_specific_output_dir = os.path.join(output_base_dir, config_label)
    os.makedirs(model_specific_output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Mean CV Accuracy
    ax1.plot(
        df_results["C"],
        df_results["mean_cv_accuracy"],
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Mean CV Accuracy",
    )
    ax1.fill_between(
        df_results["C"],
        df_results["mean_cv_accuracy"] - df_results["std_cv_accuracy"],
        df_results["mean_cv_accuracy"] + df_results["std_cv_accuracy"],
        color="tab:blue",
        alpha=0.1,
    )
    ax1.set_xlabel("C (Regularization Strength)")
    ax1.set_ylabel("Mean CV Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xscale("log")  # C values are often on a log scale
    ax1.grid(True, which="both", ls="-", alpha=0.6)

    # Create a second y-axis for Balanced Accuracy
    ax2 = ax1.twinx()
    ax2.plot(
        df_results["C"],
        df_results["mean_cv_balanced_accuracy"],
        marker="x",
        linestyle="--",
        color="tab:red",
        label="Mean CV Balanced Accuracy",
    )
    ax2.fill_between(
        df_results["C"],
        df_results["mean_cv_balanced_accuracy"]
        - df_results["std_cv_balanced_accuracy"],
        df_results["mean_cv_balanced_accuracy"]
        + df_results["std_cv_balanced_accuracy"],
        color="tab:red",
        alpha=0.1,
    )
    ax2.set_ylabel("Mean CV Balanced Accuracy", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Logistic Regression C Parameter Search Results for {config_label}")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    plot_path = os.path.join(
        model_specific_output_dir, "C_parameter_search_results.png"
    )
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"C parameter search plot saved to {plot_path}")


def plot_performance_metrics(results_df: pd.DataFrame, output_dir: str):
    """
    Plots the mean cross-validation accuracy and balanced accuracy scores for each configuration.
    This function is best suited for comparing disparate configurations using bar plots.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of all experiments.
                                   Expected to have 'nk', 'mdpk', 'nfpk', 'C', 'vars_included',
                                   'mean_cv_accuracy', and 'mean_cv_balanced_accuracy' columns.
        output_dir (str): Directory where the plot will be saved.
    """
    if results_df.empty:
        logging.warning("No results to plot performance metrics.")
        return

    logging.info("Generating performance metrics plot (bar chart summary)...")

    # Create a unique identifier for each configuration for plotting
    # Now includes C and vars_included for more comprehensive labeling
    results_df["config_id"] = results_df.apply(
        lambda row: (
            f"VARS={row['vars_included']}_"
            f"NK={int(row['nk'])}, MDPK={int(row['mdpk'])}, NFPK={int(row['nfpk'])}, C={row['C']:.4f}"
        ),
        axis=1,
    )

    # Sort by balanced accuracy for better visualization
    results_df = results_df.sort_values(by='mean_cv_balanced_accuracy', ascending=False).reset_index(drop=True)


    fig, ax1 = plt.subplots(figsize=(16, 9)) # Increased figure size for more labels

    # Plot Mean CV Accuracy
    ax1.bar(
        results_df["config_id"],
        results_df["mean_cv_accuracy"], # Use mean_cv_accuracy
        width=0.4,
        label="Mean CV Accuracy",
        align="center",
        color="skyblue",
    )
    ax1.set_ylabel("Mean CV Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)  # Accuracy typically ranges from 0 to 1

    # Create a second y-axis for Balanced Accuracy
    ax2 = ax1.twinx()
    ax2.bar(
        results_df["config_id"],
        results_df["mean_cv_balanced_accuracy"], # Use mean_cv_balanced_accuracy
        width=0.4,
        label="Mean CV Balanced Accuracy",
        align="edge",
        color="lightcoral",
        alpha=0.7,
    )
    ax2.set_ylabel("Mean CV Balanced Accuracy", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1)  # Balanced Accuracy also ranges from 0 to 1

    ax1.set_xlabel("Configuration")
    plt.title("Mean CV Accuracy and Balanced Accuracy for Different Configurations", fontsize=16)

    # Rotate x-axis labels if too many configurations
    # This block is crucial for readability with many configurations
    if len(results_df["config_id"]) > 3: # Adjust threshold as needed
        tick_locations = ax1.get_xticks()
        ax1.set_xticks(tick_locations) # Fix tick locations
        ax1.set_xticklabels(results_df["config_id"], rotation=75, ha="right", fontsize=8) # Increased rotation, smaller font
    else:
        ax1.set_xticklabels(results_df["config_id"], rotation=45, ha="right") # Still rotate for fewer

    fig.tight_layout()  # Adjust layout to prevent labels overlapping

    # Add a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    # Changed filename to be more general for multivariate context
    plot_path = os.path.join(output_dir, "multivariate_performance_metrics_summary.png")
    plt.savefig(plot_path)
    plt.close(fig)  # Close the figure to free memory
    logging.info(f"Performance metrics plot saved to {plot_path}")
    
def plot_performance_metrics_by_num_kernels(results_df: pd.DataFrame, output_dir: str):
    """
    Plots the mean cross-validation accuracy and balanced accuracy scores
    as a function of 'num_kernels', assuming other MultiRocket parameters (mdpk, nfpk)
    and C are fixed for the models in the DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of experiments.
                                   Expected to have 'nk', 'mdpk', 'nfpk', 'C', 'vars_included',
                                   'mean_cv_accuracy', 'std_cv_accuracy',
                                   'mean_cv_balanced_accuracy', 'std_cv_balanced_accuracy' columns.
        output_dir (str): Directory where the plot will be saved.
    """
    if results_df.empty:
        logging.warning("No results to plot performance metrics by num_kernels.")
        return

    logging.info("Generating performance metrics plot by num_kernels...")

    # Ensure data is sorted by num_kernels for a clean line plot
    plot_df = results_df.sort_values(by='nk').reset_index(drop=True)

    # Check if other parameters are indeed fixed
    # 'vars_included' is now explicitly part of the multivariate context
    unique_mdpk = plot_df['mdpk'].nunique()
    unique_nfpk = plot_df['nfpk'].nunique()
    unique_C = plot_df['C'].nunique()
    unique_vars = plot_df['vars_included'].nunique()

    # If any of the "fixed" parameters vary, warn the user.
    # For this plot, we specifically want to see num_kernels' effect in isolation.
    if unique_mdpk > 1 or unique_nfpk > 1 or unique_C > 1 or unique_vars > 1:
        logging.warning(
            "Multiple values found for mdpk, nfpk, C, or vars_included in the provided DataFrame for "
            "'plot_performance_metrics_by_num_kernels'. "
            "This plot assumes only 'num_kernels' varies. "
            "Plotting all data may be misleading. Consider filtering 'results_df' before calling this function "
            "to isolate a fixed set of 'mdpk', 'nfpk', and 'C' values."
        )
    
    # Extract fixed parameters for the title - take from the first row if unique, otherwise note 'Varying'
    fixed_mdpk = plot_df['mdpk'].iloc[0] if unique_mdpk == 1 else 'Varying'
    fixed_nfpk = plot_df['nfpk'].iloc[0] if unique_nfpk == 1 else 'Varying'
    fixed_C = plot_df['C'].iloc[0] if unique_C == 1 else 'Varying'
    fixed_vars = plot_df['vars_included'].iloc[0] if unique_vars == 1 else 'Varying'


    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Mean CV Accuracy with error bars
    ax1.errorbar(
        plot_df['nk'],
        plot_df['mean_cv_accuracy'],
        yerr=plot_df['std_cv_accuracy'],
        fmt='-o',        # Line with circular markers
        capsize=5,       # Size of the error bar caps
        label='Mean CV Accuracy',
        color='tab:blue',
        ecolor='lightgray', # Color of error bars
        elinewidth=1,    # Width of error bar lines
        markerfacecolor='tab:blue',
        markeredgewidth=0
    )
    ax1.set_ylabel('Mean CV Accuracy', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1)
    # Ensure all nk values are shown as ticks, and format as integer
    ax1.set_xticks(plot_df['nk']) 
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter()) 
    ax1.set_xscale('log') # num_kernels often spans orders of magnitude
    ax1.grid(True, which="both", ls="-", alpha=0.6)


    # Create a second y-axis for Mean CV Balanced Accuracy
    ax2 = ax1.twinx()
    ax2.errorbar(
        plot_df['nk'],
        plot_df['mean_cv_balanced_accuracy'],
        yerr=plot_df['std_cv_balanced_accuracy'],
        fmt='--x',       # Dashed line with 'x' markers
        capsize=5,
        label='Mean CV Balanced Accuracy',
        color='tab:red',
        ecolor='lightgray',
        elinewidth=1,
        markerfacecolor='tab:red',
        markeredgewidth=0
    )
    ax2.set_ylabel('Mean CV Balanced Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1)

    ax1.set_xlabel('Number of Kernels (num_kernels)')
    plt.title(
        f"Model Performance vs. Number of Kernels\n"
        f"(Fixed: MDPK={fixed_mdpk}, NFPK={fixed_nfpk}, C={fixed_C}, Vars='{fixed_vars}')",
        fontsize=14
    )

    # Add a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    # Changed filename to be more specific for multivariate context
    plot_path = os.path.join(output_dir, "multivariate_performance_by_num_kernels.png")
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Performance by num_kernels plot saved to {plot_path}")


def plot_classification_examples(
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_metadata_df: pd.DataFrame,
    best_model_info: dict,
    output_base_dir: str,
    vars_to_include: list,  # Pass vars_to_include to get channel names
    num_examples_per_category: int = 2,
):
    """
    Generates and saves example plots for True Positives, True Negatives,
    False Positives, and False Negatives from the best performing model.
    Each plot will show all time series channels (CPUE + oceanographic vars).

    Args:
        X_test (np.ndarray): The testing features (time series data).
        y_test (np.ndarray): The true labels for the testing data.
        X_test_metadata_df (pd.DataFrame): Metadata for the test set (Area, Alpha3_Code, Year).
        best_model_info (dict): Dictionary containing information about the best model,
                                 including 'model_path' and its configuration details.
        output_base_dir (str): The base directory for all model outputs.
        vars_to_include (list): List of strings specifying which 'Var' values were included as channels.
        num_examples_per_category (int): Number of examples to plot for each category (TP, TN, FP, FN).
    """
    if not best_model_info or best_model_info.get("status") != "Success":
        logging.warning(
            "No successful best model information available to plot classification examples."
        )
        return

    logging.info("Generating example plots for best model's predictions...")

    best_config_label = (
        f"NK={int(best_model_info['nk'])}_"
        f"MDPK={int(best_model_info['mdpk'])}_"
        f"NFPK={int(best_model_info['nfpk'])}"
    )
    model_output_dir = os.path.join(output_base_dir, best_config_label)

    # Create a subdirectory for example plots
    examples_output_dir = os.path.join(model_output_dir, "classification_examples")
    os.makedirs(examples_output_dir, exist_ok=True)

    try:
        # Load the best model
        model_path = best_model_info["model_path"]
        with open(model_path, "rb") as f:
            best_pipeline = pickle.load(f)
        logging.info(f"Successfully loaded best model from {model_path}")

        # Get predictions from the best model
        y_pred = best_pipeline.predict(X_test)

        # Identify indices for each category
        tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

        classification_categories = {
            "True Positives": tp_indices,
            "True Negatives": tn_indices,
            "False Positives": fp_indices,
            "False Negatives": fn_indices,
        }

        channel_names = ["CPUE"] + vars_to_include  # Define names for plotting legend

        # Plot examples for each category
        for category, indices in classification_categories.items():
            logging.info(
                f"Plotting examples for: {category} (found {len(indices)} instances)"
            )

            # Select random examples, up to num_examples_per_category
            if len(indices) > num_examples_per_category:
                selected_indices = np.random.choice(
                    indices, num_examples_per_category, replace=False
                )
            else:
                selected_indices = indices  # Take all if fewer than desired examples

            for i, idx in enumerate(selected_indices):
                # ts_data is now (n_channels, n_timestamps)
                ts_data_all_channels = X_test[idx, :, :]
                true_label = y_test[idx]
                predicted_label = y_pred[idx]
                metadata = X_test_metadata_df.iloc[idx]

                plt.figure(figsize=(12, 7))

                for channel_idx, channel_ts in enumerate(ts_data_all_channels):
                    plt.plot(channel_ts, label=channel_names[channel_idx])

                plt.title(
                    f"{category} Example {i+1}: Area={metadata['Area']}, Species={metadata['Alpha3_Code']},"
                    f" Year={metadata['Year']}\nTrue: {int(true_label)}, Predicted: {int(predicted_label)}"
                )
                plt.xlabel("Time Step (Months since 1993-01)")
                plt.ylabel("Value")
                plt.grid(True)
                plt.legend(title="Time Series Channel")

                # Determine plot background color based on correctness
                if category.startswith("True"):
                    plt.gca().set_facecolor("#e6ffe6")  # Light green for correct
                else:
                    plt.gca().set_facecolor("#ffe6e6")  # Light red for incorrect

                plot_filename = os.path.join(
                    examples_output_dir,
                    f"mv_{category.replace(' ', '_').lower()}_example_{i+1}.png",
                )
                plt.savefig(plot_filename)
                plt.close()
                logging.debug(f"Saved {category} example {i+1} plot to {plot_filename}")

    except Exception as e:
        logging.error(
            f"Error generating classification example plots: {e}", exc_info=True
        )


def plot_oceanography_correlation(oceanography_df: pd.DataFrame, output_dir: str):
    """
    Calculates and plots the Pearson correlation matrix of different oceanographic variables
    over time.

    Args:
        oceanography_df (pd.DataFrame): DataFrame with 'time' (datetime), 'Value',
            'Var' (string, e.g., 'SST', 'Salinity'), and 'Area' (int) columns.
        output_dir (str): Directory where the correlation matrix plot will be saved.
    """
    logging.info(
        "Calculating and plotting oceanography variable Pearson correlation matrix..."
    )

    if oceanography_df.empty:
        logging.warning(
            "Oceanography DataFrame is empty. Cannot plot correlation matrix."
        )
        return

    # Group by time and variable, then take the mean value for that time-variable combination
    # across all areas. This creates a representative time series for each variable.
    ocean_pivot_df = (
        oceanography_df.groupby(["time", "Var"])["Value"].mean().unstack(level="Var")
    )

    # Fill any potential missing values. For correlation, NaNs will result in NaNs in the matrix.
    # It's crucial to handle them. Interpolation might be more appropriate than filling with 0
    # if you want to preserve trends, but 0 is a simple, safe default if you're unsure.
    # We'll use linear interpolation for within-series gaps, and then 0 for any remaining NaNs (e.g., at ends).
    ocean_pivot_df = ocean_pivot_df.interpolate(method="linear", axis=0).fillna(0)

    logging.info(
        f"Pivoted oceanography data shape for correlation: {ocean_pivot_df.shape}"
    )
    logging.debug(
        f"Pivoted oceanography data columns: {ocean_pivot_df.columns.tolist()}"
    )

    if ocean_pivot_df.empty or ocean_pivot_df.shape[1] < 2:
        logging.warning(
            "Not enough variables or data points after pivoting to compute correlation matrix."
        )
        return

    # Calculate the Pearson correlation matrix
    correlation_matrix = ocean_pivot_df.corr(method="pearson")

    logging.info("Pearson correlation matrix calculated.")
    logging.debug(f"Correlation matrix shape: {correlation_matrix.shape}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Annotate the heatmap with the correlation values
        fmt=".2f",  # Format annotations to two decimal places
        cmap="coolwarm",  # Color map suitable for -1 to 1 range (e.g., "coolwarm", "RdBu", "vlag")
        linewidths=0.5,  # Lines between cells
        linecolor="black",  # Color of the lines
        cbar_kws={
            "label": "Pearson Correlation Coefficient"
        },  # Label for the color bar
    )
    plt.title("Pearson Correlation Matrix of Oceanographic Variables", fontsize=16)
    plt.xlabel("Oceanographic Variable", fontsize=12)
    plt.ylabel("Oceanographic Variable", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjust layout to prevent labels overlapping

    plot_path = os.path.join(output_dir, "oceanography_pearson_correlation_matrix.png")
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    logging.info(f"Oceanography Pearson correlation matrix plot saved to {plot_path}")


def main(args):
    logging.info(
        "--- Starting CPUE & Oceanography Multivariate Classifier Workflow ---"
    )

    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "model_data")

    # Create a base output directory for all runs
    output_base_dir = os.path.join(parent_dir, "multivariate_model_output_runs")
    os.makedirs(output_base_dir, exist_ok=True)

    # Define the path for the consolidated results CSV
    results_file = os.path.join(
        output_base_dir, "multivariate_model_performance_summary.csv"
    )

    logging.info(f"Loading data from: {input_dir}")
    try:
        with open(os.path.join(input_dir, "cpue.pkl"), "rb") as file:
            cpue_df = pickle.load(file)
        with open(os.path.join(input_dir, "target.pkl"), "rb") as file:
            target_df = pickle.load(file)
        with open(os.path.join(input_dir, "oceanography.pkl"), "rb") as file:
            oceanography_df = pickle.load(file)

        # Ensure 'time' column is datetime
        if "time" in oceanography_df.columns:
            oceanography_df["time"] = pd.to_datetime(oceanography_df["time"])

        logging.info("Data loaded successfully.")
    except FileNotFoundError as e:
        logging.error(
            f"Error: Required data file not found in '{input_dir}': {e}. "
            "Please ensure 'cpue.pkl', 'target.pkl', and 'oceanography.pkl' exist."
        )
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred loading data: {e}", exc_info=True)
        return

    plot_oceanography_correlation(oceanography_df, output_base_dir)

    # Define which oceanographic variables to include
    vars_to_include = ["thetao_mean", "nppv", "spco2", "no3"]
    logging.info(f"Oceanographic variables to include: {vars_to_include}")

    logging.info("Formatting data and splitting into train/test sets...")
    try:
        # Pass oceanography_df and vars_to_include to format_data
        X_train, X_test, y_train, y_test, X_test_metadata_df = format_data(
            cpue_df, target_df, oceanography_df, vars_to_include
        )
    except ValueError as e:
        logging.error(f"Error during data formatting: {e}")
        return
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during data formatting: {e}", exc_info=True
        )
        return

    all_experiment_results = []

    # If specific args are provided, run only that single configuration.
    # Otherwise, run a grid search (as defined below).
    if args.num_kernels and args.max_dilations and args.n_features:
        if args.c_search:
            logging.info(
                "MultiRocket parameters specified. Running GridSearchCV for Logistic Regression C parameter."
            )
            rocket_fixed_config = {
                "num_kernels": args.num_kernels,
                "max_dilations": args.max_dilations,
                "n_features": args.n_features,
            }

            C_values_to_search = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]

            c_search_results_for_config = run_c_grid_search(
                X_train,
                X_test,
                y_train,
                y_test,
                rocket_fixed_config,
                output_base_dir,
                C_values_to_search,
            )
            if c_search_results_for_config:
                # all_experiment_results.extend(c_search_results_for_config)
                print(c_search_results_for_config)
                # Plot the C search results specifically for this config
                plot_c_search_results(
                    c_search_results_for_config, rocket_fixed_config, output_base_dir
                )
        elif args.var_search:
            # Define fixed MultiRocket parameters for this variable search
            fixed_rocket_config_for_vars_search = {
                "num_kernels": args.num_kernels,
                "max_dilations": args.max_dilations,
                "n_features": args.n_features,
            }
            # Fixed C value for the variable combination search
            fixed_C_for_vars_search = 0.01

            vars_search_results = run_vars_grid_search(
                cpue_df,
                target_df,
                oceanography_df,
                vars_to_include,
                fixed_rocket_config_for_vars_search,
                fixed_C_for_vars_search,
                output_base_dir,
            )
            if vars_search_results:
                all_experiment_results.extend(vars_search_results)
                vars_search_results_df = pd.DataFrame(vars_search_results)
                plot_vars_search_results(
                    vars_search_results_df,
                    fixed_rocket_config_for_vars_search,
                    fixed_C_for_vars_search,
                    output_base_dir,
                )
        else:
            logging.info(
                "Running single configuration as specified by command-line arguments."
            )
            config = {
                "num_kernels": args.num_kernels,
                "max_dilations": args.max_dilations,
                "n_features": args.n_features,
                "C": 0.01
            }
            result = run_experiment(
                X_train, X_test, y_train, y_test, config, output_base_dir
            )
            if result:
                all_experiment_results.append(result)
    else:
        logging.info(
            "No specific configuration provided. Running a predefined grid search."
        )
        # Define parameter grids for MultiRocket
        num_kernels_options = [100, 500, 1000, 5000, 10000]
        mdpk = 16
        nfpk = 4

        for nk in num_kernels_options:
            config = {
                "num_kernels": nk,
                "max_dilations": mdpk,
                "n_features": nfpk,
            }
            result = run_experiment(
                X_train, X_test, y_train, y_test, config, output_base_dir
            )
            if result:
                all_experiment_results.append(result)
            gc.collect()  # Trigger garbage collection after each run
        
        if all_experiment_results:
            rocket_search_df = pd.DataFrame(all_experiment_results)
            # Filter for successful runs and those matching fixed mdpk, nfpk, C
            # In this 'else' block, mdpk, nfpk, C are already fixed in the loop's config
            filtered_for_nk_plot = rocket_search_df[
                (rocket_search_df['mdpk'] == mdpk) &
                (rocket_search_df['nfpk'] == nfpk) &
                (rocket_search_df['status'] == 'Success')
            ].copy()

            if not filtered_for_nk_plot.empty and filtered_for_nk_plot['nk'].nunique() > 1:
                logging.info("Calling plot_performance_metrics_by_num_kernels...")
                plot_performance_metrics_by_num_kernels(filtered_for_nk_plot, output_base_dir)
            else:
                logging.info("Not enough varying 'num_kernels' values or successful runs to generate 'performance_by_num_kernels' plot.")


    # Log all results to a single CSV
    if all_experiment_results:
        final_results_df = pd.DataFrame(all_experiment_results)

        # Check if file exists to handle header appropriately
        if not os.path.exists(results_file):
            final_results_df.to_csv(results_file, mode="w", header=True, index=False)
            logging.info(f"Created new results file at: {results_file}")
        else:
            # Filter out existing configurations if they were already run and successful
            try:
                existing_results_df = pd.read_csv(results_file)
                id_cols = [
                    "nk",
                    "mdpk",
                    "nfpk",
                    "C",
                    "vars_included",
                ]  # Include vars_included in ID

                existing_configs_status = existing_results_df[
                    id_cols + ["status"]
                ].drop_duplicates()

                new_results_to_add = []
                for res in all_experiment_results:
                    # Convert vars_included list to string for consistent comparison if needed
                    res_vars_str = (
                        ", ".join(sorted(res["vars_included"].split(", ")))
                        if isinstance(res["vars_included"], str)
                        else res["vars_included"]
                    )

                    match = existing_configs_status[
                        (existing_configs_status["nk"] == res["nk"])
                        & (existing_configs_status["mdpk"] == res["mdpk"])
                        & (existing_configs_status["nfpk"] == res["nfpk"])
                        & (existing_configs_status["C"] == res["C"])
                        & (existing_configs_status["vars_included"] == res_vars_str)
                        & (existing_configs_status["status"] == "Success")
                    ]

                    if not match.empty:
                        logging.info(
                            f"Skipping addition of already logged (and successful) config: {res_vars_str}, NK={res['nk']}, C={res['C']}"
                        )
                    else:
                        new_results_to_add.append(res)

                if new_results_to_add:
                    pd.DataFrame(new_results_to_add).to_csv(
                        results_file, mode="a", header=False, index=False
                    )
                    logging.info(
                        f"Appended {len(new_results_to_add)} new results to {results_file}"
                    )
                else:
                    logging.info("No new unique results to append to the summary file.")

            except Exception as e:
                logging.error(
                    f"Error while trying to append results to {results_file}: {e}. Appending anyway.",
                    exc_info=True,
                )
                pd.DataFrame(all_experiment_results).to_csv(
                    results_file, mode="a", header=False, index=False
                )

        # Plot performance metrics for all successful runs
        plot_df = final_results_df[final_results_df["status"] == "Success"].copy()
        plot_performance_metrics(plot_df, output_base_dir)

        # Find the best model and plot classification examples
        if not plot_df.empty:
            best_model_row = plot_df.loc[plot_df["balanced_accuracy"].idxmax()]
            best_model_info = best_model_row.to_dict()
            logging.info(
                f"Best model (by balanced accuracy) found: NK={best_model_info['nk']}, MDPK={best_model_info['mdpk']}, NFPK={best_model_info['nfpk']}"
            )

            # Call the new plotting function for classification examples
            plot_classification_examples(
                X_test,
                y_test,
                X_test_metadata_df,
                best_model_info,
                output_base_dir,
                vars_to_include,  # Pass channel names for plotting
            )
        else:
            logging.warning(
                "No successful runs found to determine the best model for example plots."
            )
    else:
        logging.info("No experiment results to save.")

    logging.info(
        "--- CPUE & Oceanography Multivariate Classifier Workflow Complete ---"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CPUE & Oceanography Multivariate Classifier Workflow. "
        "Provide --num_kernels, --max_dilations, --n_features to run a single configuration. "
        "Provide --c_search if you want to perform a grid search over logistic regression parameter C. "
        "Provide --var_search if you want to perform a search over the subsets of the oceanographic variables as inputs. "
        "If none are provided, a predefined grid search will be executed."
    )
    # Make arguments optional for grid search, but still allow single run
    parser.add_argument(
        "--num_kernels", type=int, help="Number of kernels for MultiRocket."
    )
    parser.add_argument("--max_dilations", type=int, help="Max dilations per kernel.")
    parser.add_argument("--n_features", type=int, help="Number of features per kernel.")
    parser.add_argument("--c_search", type=bool, help="Perform grid search over C.")
    parser.add_argument("--var_search", type=bool, help="Perform search over subsets of oceanographic variables.")

    parsed_args = parser.parse_args()
    main(parsed_args)
