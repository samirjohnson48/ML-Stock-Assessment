import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from math import floor
from itertools import product
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

def format_data(cpue_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple:
    """
    Formats and splits data for model training and testing.

    This function merges CPUE timeseries data with target data, adjusts the CPUE
    timeseries length, and splits the data into training and testing sets.

    Args:
        cpue_df (pd.DataFrame): DataFrame containing CPUE timeseries data with
            "Area", "Species", "Year", and 'CPUE' columns. The 'CPUE' column
            should contain the actual time series as a list or array.
        target_df (pd.DataFrame): DataFrame containing target data with "Area",
            "Alpha3_Code", and "S" (target variable) columns.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test, which are
            the training and testing data for the model. X_train and X_test are
            NumPy arrays of shape (samples, 1, timeseries_length) and type float32.
            y_train and y_test are NumPy arrays of shape (samples,) containing the
            target variable.
    """
    logging.info("Starting data formatting...")
    # Use clearer variable names and ensure robust merging
    model_data = pd.merge(cpue_df, target_df, on=["Area", "Alpha3_Code"], how="inner")
    logging.info(f"Merged data contains {len(model_data)} records.")

    # Determine dynamic year range from CPUE columns
    years_in_cpue_cols = [
        col
        for col in cpue_df.columns
        if isinstance(col, int)
        and col >= 1950
        and col <= 2023  # Kept boundary for sanity
    ]
    if not years_in_cpue_cols:
        raise ValueError(
            "No valid year columns found in cpue_df for time series extraction."
        )

    min_cpue_year = min(years_in_cpue_cols)
    max_cpue_year = max(years_in_cpue_cols)

    logging.info(
        f"Dynamic CPUE year range detected: {min_cpue_year} to {max_cpue_year}."
    )

    def create_cpue_input(row: pd.Series) -> np.ndarray:
        """Helper function to extract CPUE time series based on year range."""
        y_end = row["Year"]
        # Use a consistent window length based on max available years or define fixed
        # For a fixed window of, say, 74 years (2023-1950+1):
        window_length = (
            max_cpue_year - min_cpue_year + 1
        )  # Dynamic window length based on detected range
        y_start = y_end - window_length + 1

        # Ensure we only pick years that are actual columns in the DataFrame
        cpue_ts_years = [
            year
            for year in range(y_start, y_end + 1)
            if year in years_in_cpue_cols
            and year in row.index  # Ensure year is in both detected and row index
        ]

        if not cpue_ts_years:
            return np.array([])

        return row[cpue_ts_years].values.astype(np.float32)

    model_data["CPUE"] = model_data.apply(create_cpue_input, axis=1)

    initial_samples = len(model_data)
    model_data = model_data[model_data["CPUE"].apply(lambda x: x.size > 0)]
    logging.info(
        f"Filtered out {initial_samples - len(model_data)} records with empty CPUE time series."
    )

    if model_data.empty:
        raise ValueError(
            "No valid time series data found after formatting. Check 'cpue_df' and 'target_df' contents and year range logic."
        )

    max_ts_length = max(len(ts) for ts in model_data["CPUE"])
    logging.info(f"Maximum time series length found: {max_ts_length}.")

    X = np.array(
        [
            np.pad(ts, (0, max_ts_length - len(ts)), "constant")
            for ts in model_data["CPUE"]
        ]
    ).astype(np.float32)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    y = model_data["S"].values.astype(np.float32)
    
    return X, y, model_data


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


def run_experiment(X_train_cv: np.array, y_train_cv: np.array, X_test_holdout: np.array, y_test_holdout: np.array, config: dict, output_base_dir: str):
    """
    Runs a single model training and evaluation experiment for a given configuration.
    Performs K-Fold Cross-Validation on X_train_cv/y_train_cv.
    Trains a final model on X_train_cv/y_train_cv.
    Evaluates the final model and plots confusion matrix on X_test_holdout/y_test_holdout.

    Args:
        X_train_cv (np.ndarray): The training feature set for cross-validation and final model training.
        y_train_cv (np.ndarray): The training target set for cross-validation and final model training.
        X_test_holdout (np.ndarray): The holdout test feature set for final, unbiased evaluation.
        y_test_holdout (np.ndarray): The holdout test target set for final, unbiased evaluation.
        config (dict): Dictionary containing MultiRocket parameters (num_kernels,
                       max_dilations, n_features) and potentially C.
        output_base_dir (str): The base directory where all experiment outputs will be stored.

    Returns:
        dict: A dictionary of results for the current configuration, or None if an error occurred.
    """
    nk = config.get("num_kernels")
    mdpk = config.get("max_dilations")
    nfpk = config.get("n_features")
    C_val = config.get("C", 0.01) # Default C if not specified

    
    config_label = f"NK={nk}_MDPK={mdpk}_NFPK={nfpk}_C={C_val}"

    model_specific_output_dir = os.path.join(output_base_dir, config_label)
    os.makedirs(model_specific_output_dir, exist_ok=True)
    
    logging.info(f"--- Starting experiment for configuration: {config_label} ---")
    
    current_encoder_args = {
        "num_kernels": nk,
        "max_dilations_per_kernel": mdpk,
        "n_features_per_kernel": nfpk,
        "normalise": True,
        "n_jobs": 1, # Set to 1 for internal MultiRocket parallelism when cross_validate handles outer parallelism
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
        'mean_cv_accuracy': np.nan,
        'std_cv_accuracy': np.nan,
        'mean_cv_balanced_accuracy': np.nan,
        'std_cv_balanced_accuracy': np.nan,
        'holdout_test_accuracy': np.nan, # New field for holdout test score
        'holdout_test_balanced_accuracy': np.nan, # New field for holdout test score
        'training_time_seconds': np.nan,
        'confusion_matrix_path': None, 
        'model_path': None,
        'status': 'Failed',
        'vars_included': 'CPUE_only'
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
        
        # Define the cross-validation strategy
        cv_folds = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score)
        }

        # Perform cross-validation on the provided X_train_cv, y_train_cv
        cv_results = cross_validate(
            current_pipeline,
            X_train_cv, y_train_cv,
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
        mean_acc = np.mean(cv_results['test_accuracy'])
        std_acc = np.std(cv_results['test_accuracy'])
        mean_b_acc = np.mean(cv_results['test_balanced_accuracy'])
        std_b_acc = np.std(cv_results['test_balanced_accuracy'])

        logging.info(f"  Mean CV Accuracy: {mean_acc:.4f} (Std: {std_acc:.4f})")
        logging.info(f"  Mean CV Balanced Accuracy: {mean_b_acc:.4f} (Std: {std_b_acc:.4f})\n")
        results_data['mean_cv_accuracy'] = mean_acc
        results_data['std_cv_accuracy'] = std_acc
        results_data['mean_cv_balanced_accuracy'] = mean_b_acc
        results_data['std_cv_balanced_accuracy'] = std_b_acc
        
        # Train the final model on the entire training dataset (X_train_cv, y_train_cv)
        logging.info("Training final model on full training dataset for holdout evaluation and saving...")
        final_model_start_time = time()
        current_pipeline.fit(X_train_cv, y_train_cv) # Fit on the full training data
        final_model_train_time = time() - final_model_start_time
        logging.info(f"Final model training time: {final_model_train_time:.2f} seconds")

        # Evaluate the final trained model on the separate holdout test set
        logging.info("Evaluating final model on separate holdout test set...")
        y_pred_holdout = current_pipeline.predict(X_test_holdout)

        holdout_acc = accuracy_score(y_test_holdout, y_pred_holdout)
        holdout_b_acc = balanced_accuracy_score(y_test_holdout, y_pred_holdout)
        
        logging.info(f"  Holdout Test Accuracy: {holdout_acc:.4f}")
        logging.info(f"  Holdout Test Balanced Accuracy: {holdout_b_acc:.4f}\n")
        results_data['holdout_test_accuracy'] = holdout_acc
        results_data['holdout_test_balanced_accuracy'] = holdout_b_acc

        # --- Print and Plot Confusion Matrix for the final model on the holdout test data ---
        cm = confusion_matrix(y_test_holdout, y_pred_holdout)
        logging.info(f"Confusion Matrix (from final model on holdout test data):\n%s", cm)

        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {config_label}\n(Evaluated on Holdout Test Set)", fontsize=14)

        plot_path = os.path.join(model_specific_output_dir, "confusion_matrix_holdout_test_data.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Confusion Matrix plot saved to {plot_path}")
        results_data['confusion_matrix_path'] = plot_path # Store this path in results

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

    config_label = f"NK={nk}_MDPK={mdpk}_NFPK={nfpk}_C_Search" # Modified label for C search
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
            n_splits=10, shuffle=True, random_state=42 # Changed to 10 folds
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
    for i, C_val_result in enumerate(cv_results["param_logisticregression__C"]):
        res_dict = {
            "nk": nk,
            "mdpk": mdpk,
            "nfpk": nfpk,
            "C": C_val_result,
            "mean_cv_accuracy": cv_results[f"mean_test_accuracy"][i],
            "std_cv_accuracy": cv_results[f"std_test_accuracy"][i],
            "mean_cv_balanced_accuracy": cv_results[f"mean_test_balanced_accuracy"][i],
            "std_cv_balanced_accuracy": cv_results[f"std_test_balanced_accuracy"][i],
            "test_accuracy_best_model": (
                test_acc
                if C_val_result == grid_search.best_params_["logisticregression__C"]
                else np.nan
            ),
            "test_balanced_accuracy_best_model": (
                test_b_acc
                if C_val_result == grid_search.best_params_["logisticregression__C"]
                else np.nan
            ),
            "training_time_seconds": tot_time,  # Total time for grid search
            "model_path": (
                best_model_path
                if C_val_result == grid_search.best_params_["logisticregression__C"]
                else None
            ),
            "status": "Success",  # Assume success if grid search completed
            'vars_included': 'CPUE_only' # Indicate that only CPUE was used
        }
        c_search_results.append(res_dict)

    logging.info(
        f"--- GridSearchCV for C on configuration: {config_label} Complete ---"
    )
    return c_search_results


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
    Plots the accuracy and balanced accuracy scores for each configuration.
    This function is best suited for comparing disparate configurations using bar plots.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of all experiments.
                                   Expected to have 'nk', 'mdpk', 'nfpk', 'C',
                                   'mean_cv_accuracy', and 'mean_cv_balanced_accuracy' columns.
        output_dir (str): Directory where the plot will be saved.
    """
    if results_df.empty:
        logging.warning("No results to plot performance metrics.")
        return

    logging.info("Generating performance metrics plot (bar chart summary)...")

    # Create a unique identifier for each configuration for plotting
    results_df["config_id"] = results_df.apply(
        lambda row: f"NK={int(row['nk'])}, MDPK={int(row['mdpk'])}, NFPK={int(row['nfpk'])}, C={row['C']}",
        axis=1,
    )

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Mean CV Accuracy
    ax1.bar(
        results_df["config_id"],
        results_df["mean_cv_accuracy"],
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
        results_df["mean_cv_balanced_accuracy"],
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
    plt.title("Mean CV Accuracy and Balanced Accuracy for Different Configurations")

    # Rotate x-axis labels if too many configurations
    if len(results_df["config_id"]) > 3:
        tick_locations = ax1.get_xticks()
        ax1.set_xticks(tick_locations)
        ax1.set_xticklabels(results_df["config_id"], rotation=45, ha="right")

    fig.tight_layout()  # Adjust layout to prevent labels overlapping

    # Add a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    plot_path = os.path.join(output_dir, "performance_metrics_summary.png")
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
                                   Expected to have 'nk', 'mdpk', 'nfpk', 'C',
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
    # 'vars_included' will always be 'CPUE_only' in this univariate workflow, but included for consistency
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

    plot_path = os.path.join(output_dir, "performance_by_num_kernels.png")
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Performance by num_kernels plot saved to {plot_path}")


def plot_classification_examples(
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_metadata_df: pd.DataFrame,
    best_model_info: dict,
    output_base_dir: str,
    num_examples_per_category: int = 2,
):
    """
    Generates and saves example plots for True Positives, True Negatives,
    False Positives, and False Negatives from the best performing model.

    Args:
        X_test (np.ndarray): The testing features (time series data).
        y_test (np.ndarray): The true labels for the testing data.
        X_test_metadata_df (pd.DataFrame): Metadata for the test set (Area, Alpha3_Code, Year).
        best_model_info (dict): Dictionary containing information about the best model,
                                 including 'model_path' and its configuration details.
        output_base_dir (str): The base directory for all model outputs.
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
        f"NFPK={int(best_model_info['nfpk'])}_"
        f"C={best_model_info['C']}" # Include C in label
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
        # True Positives: Actual 1, Predicted 1
        tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        # True Negatives: Actual 0, Predicted 0
        tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
        # False Positives: Actual 0, Predicted 1 (Type I error)
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        # False Negatives: Actual 1, Predicted 0 (Type II error)
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

        classification_categories = {
            "True Positives": tp_indices,
            "True Negatives": tn_indices,
            "False Positives": fp_indices,
            "False Negatives": fn_indices,
        }

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
                ts_data = X_test[idx, 0, :]  # Extract the single time series
                true_label = y_test[idx]
                predicted_label = y_pred[idx]
                metadata = X_test_metadata_df.iloc[idx]

                plt.figure(figsize=(10, 5))
                plt.plot(ts_data)
                plt.title(
                    f"{category} Example {i+1}: Area={metadata['Area']}, Species={metadata['Alpha3_Code']},"
                    f" Year={metadata['Year']}\nTrue: {int(true_label)}, Predicted: {int(predicted_label)}"
                )
                plt.xlabel("Time Step (Relative Year)")
                plt.ylabel("CPUE Value")
                plt.grid(True)

                # Determine plot color based on correctness
                if category.startswith("True"):
                    plt.gca().set_facecolor("#e6ffe6")  # Light green for correct
                else:
                    plt.gca().set_facecolor("#ffe6e6")  # Light red for incorrect

                plot_filename = os.path.join(
                    examples_output_dir,
                    f"{category.replace(' ', '_').lower()}_example_{i+1}.png",
                )
                plt.savefig(plot_filename)
                plt.close()
                logging.debug(f"Saved {category} example {i+1} plot to {plot_filename}")

    except Exception as e:
        logging.error(
            f"Error generating classification example plots: {e}", exc_info=True
        )


def main(args):
    logging.info("--- Starting CPUE Classifier Workflow ---")

    parent_dir = os.getcwd()
    input_dir = os.path.join(parent_dir, "model_data")

    # Create a base output directory for all runs
    output_base_dir = os.path.join(parent_dir, "univariate_model_output_runs")
    os.makedirs(output_base_dir, exist_ok=True)

    # Define the path for the consolidated results CSV
    results_file = os.path.join(output_base_dir, "cpue_model_performance_summary.csv")

    logging.info(f"Loading data from: {input_dir}")
    try:
        # Placeholder for data loading - replace with actual paths/logic
        # For demonstration, creating dummy dataframes if files don't exist
        cpue_df_path = os.path.join(input_dir, "cpue.pkl")
        target_df_path = os.path.join(input_dir, "target.pkl")

        if os.path.exists(cpue_df_path) and os.path.exists(target_df_path):
            with open(cpue_df_path, "rb") as file:
                cpue_df = pickle.load(file)
            with open(target_df_path, "rb") as file:
                target_df = pickle.load(file)
            logging.info("Data loaded successfully from pickle files.")
        else:
            logging.warning("Data pickle files not found. Creating dummy data for demonstration.")
            # Create dummy cpue_df
            cpue_data = {
                'Area': [f'Area{i}' for i in range(100)],
                'Alpha3_Code': [f'SPECIES{i%5}' for i in range(100)],
                'Year': [1980 + i for i in range(100)], # Example start year
            }
            # Add dynamic year columns for CPUE time series
            for yr in range(1950, 2024):
                cpue_data[yr] = np.random.rand(100) * 100 # Random CPUE values
            cpue_df = pd.DataFrame(cpue_data)
            
            # Create dummy target_df
            target_data = {
                'Area': [f'Area{i}' for i in range(100)],
                'Alpha3_Code': [f'SPECIES{i%5}' for i in range(100)],
                'S': np.random.randint(0, 2, 100) # Binary target (0 or 1)
            }
            target_df = pd.DataFrame(target_data)
            
            # Ensure some overlap in Area/Alpha3_Code for merge
            cpue_df = cpue_df.head(50) # Use subset for merge to guarantee common entries
            target_df = target_df.head(50)


    except Exception as e:
        logging.error(f"An unexpected error occurred loading or creating data: {e}", exc_info=True)
        return

    logging.info("Formatting data and splitting into train/test sets...")
    try:
        X, y, model_data = format_data(cpue_df, target_df)
        
        logging.info(f"Splitting data into training (80%) and testing (20%) sets.")
        # Perform the train-test split once here
        X_train, X_test, y_train, y_test, _, X_test_metadata_df = train_test_split(
            X,
            y,
            model_data[["Area", "Alpha3_Code", "Year"]],
            test_size=0.2,
            random_state=42,
            stratify=y, # Important for balanced classes in classification
        )
        logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logging.info("Data formatting complete.")
    except ValueError as e:
        logging.error(f"Error during data formatting: {e}")
        return
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during data formatting: {e}", exc_info=True
        )
        return

    all_experiment_results = []
    best_model_overall_info = None
    best_balanced_accuracy_overall = -1.0 # Initialize with a low value

    # If specific MultiRocket args are provided (and optionally C), run C search or single experiment.
    # Otherwise, run the predefined MultiRocket grid search.
    if args.num_kernels and args.max_dilations and args.n_features:
        rocket_fixed_config = {
            "num_kernels": args.num_kernels,
            "max_dilations": args.max_dilations,
            "n_features": args.n_features,
        }

        if args.C is not None:
            logging.info("Running a single experiment with specified MultiRocket and C parameters.")
            # Call run_experiment with the fixed C value
            config_for_single_run = {**rocket_fixed_config, "C": args.C}
            result = run_experiment(X_train, y_train, X_test, y_test, config_for_single_run, output_base_dir) # run_experiment uses X, y for CV
            if result:
                all_experiment_results.append(result)
                if result['status'] == 'Success' and result['mean_cv_balanced_accuracy'] > best_balanced_accuracy_overall:
                    best_balanced_accuracy_overall = result['mean_cv_balanced_accuracy']
                    best_model_overall_info = result
            else:
                logging.error(f"Single experiment failed for config: {config_for_single_run}")
        else:
            logging.info(
                "MultiRocket parameters specified without C. Running GridSearchCV for Logistic Regression C parameter."
            )
            # C values to search, <= 0.1
            C_values_to_search = [1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

            c_search_results_for_config = run_c_grid_search(
                X_train, # Use X_train, y_train for GridSearchCV
                X_test,
                y_train,
                y_test,
                rocket_fixed_config,
                output_base_dir,
                C_values_to_search,
            )
            if c_search_results_for_config:
                all_experiment_results.extend(c_search_results_for_config)
                # Plot the C search results specifically for this config
                plot_c_search_results(
                    c_search_results_for_config, rocket_fixed_config, output_base_dir
                )
                # Find the best model from the C search results
                c_search_df = pd.DataFrame(c_search_results_for_config)
                successful_c_runs = c_search_df[c_search_df['status'] == 'Success']
                if not successful_c_runs.empty:
                    current_best_c_model = successful_c_runs.loc[successful_c_runs['mean_cv_balanced_accuracy'].idxmax()]
                    if current_best_c_model['mean_cv_balanced_accuracy'] > best_balanced_accuracy_overall:
                        best_balanced_accuracy_overall = current_best_c_model['mean_cv_balanced_accuracy']
                        best_model_overall_info = current_best_c_model.to_dict()
            else:
                logging.warning("C grid search returned no results.")

    else:
        logging.info(
            "No specific configuration provided. Running a predefined MultiRocket grid search with fixed C."
        )
        # Define parameter grids for MultiRocket
        num_kernels_options = [100, 500, 1000, 5000, 10000]
        max_dilations_options = [16] # Fixed for this loop
        n_features_options = [4]    # Fixed for this loop
        fixed_C_for_rocket_search = 0.01 # Fixed C for these MultiRocket experiments

        configs_to_run = []
        for nk, mdpk, nfpk in product(
            num_kernels_options, max_dilations_options, n_features_options
        ):
            configs_to_run.append({
                "num_kernels": nk,
                "max_dilations": mdpk,
                "n_features": nfpk,
                "C": fixed_C_for_rocket_search # Add fixed C to config
            })

        for config in configs_to_run:
            result = run_experiment(
                X_train, y_train, X_test, y_test, config, output_base_dir
            ) # run_experiment uses X, y for CV
            if result:
                all_experiment_results.append(result)
                if result['status'] == 'Success' and result['mean_cv_balanced_accuracy'] > best_balanced_accuracy_overall:
                    best_balanced_accuracy_overall = result['mean_cv_balanced_accuracy']
                    best_model_overall_info = result
            gc.collect()  # Trigger garbage collection after each run

        # After the MultiRocket grid search, plot performance by num_kernels
        if all_experiment_results:
            rocket_search_df = pd.DataFrame(all_experiment_results)
            # Filter for successful runs and those matching fixed mdpk, nfpk, C
            # In this 'else' block, mdpk, nfpk, C are already fixed in the loop's config
            filtered_for_nk_plot = rocket_search_df[
                (rocket_search_df['mdpk'] == max_dilations_options[0]) &
                (rocket_search_df['nfpk'] == n_features_options[0]) &
                (rocket_search_df['C'] == fixed_C_for_rocket_search) &
                (rocket_search_df['status'] == 'Success')
            ].copy()

            if not filtered_for_nk_plot.empty and filtered_for_nk_plot['nk'].nunique() > 1:
                logging.info("Calling plot_performance_metrics_by_num_kernels...")
                plot_performance_metrics_by_num_kernels(filtered_for_nk_plot, output_base_dir)
            else:
                logging.info("Not enough varying 'num_kernels' values or successful runs to generate 'performance_by_num_kernels' plot.")


    # Log all results to a single CSV (consolidated results from all scenarios)
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
                # Define a unique identifier for configurations (nk, mdpk, nfpk, C, vars_included)
                # This makes the check more robust if you ever introduce other varying parameters
                existing_configs_identifiers = existing_results_df[
                    ["nk", "mdpk", "nfpk", "C", "vars_included"]
                ].drop_duplicates()

                new_results_to_add = []
                for res in all_experiment_results:
                    current_config_identifier = pd.DataFrame(
                        [{
                            "nk": res["nk"],
                            "mdpk": res["mdpk"],
                            "nfpk": res["nfpk"],
                            "C": res["C"],
                            "vars_included": res["vars_included"]
                        }]
                    )
                    # Check if this exact config already exists in the existing results
                    if not current_config_identifier.merge(
                        existing_configs_identifiers, on=["nk", "mdpk", "nfpk", "C", "vars_included"], how="inner"
                    ).empty:
                        logging.info(
                            f"Skipping addition of already logged config: NK={res['nk']}, MDPK={res['mdpk']}, NFPK={res['nfpk']}, C={res['C']}"
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

        # Always plot overall performance metrics (bar chart) if there are any results
        successful_runs_df = final_results_df[
            final_results_df["status"] == "Success"
        ].copy()
        
        # Ensure 'accuracy' and 'balanced_accuracy' columns exist for the original plot_performance_metrics
        # For cross-validation results, these are mean_cv_accuracy and mean_cv_balanced_accuracy
        # Let's adjust plot_performance_metrics to use the CV means
        plot_performance_metrics(successful_runs_df, output_base_dir) # This plot now uses mean_cv_accuracy/balanced_accuracy

        if best_model_overall_info:
            logging.info(
                f"Best overall model (by balanced accuracy) found: NK={best_model_overall_info['nk']}, "
                f"MDPK={best_model_overall_info['mdpk']}, NFPK={best_model_overall_info['nfpk']}, C={best_model_overall_info['C']}"
            )

            # Call the classification examples plot for the single best model
            # Pass the initial test set (X_test, y_test, X_test_metadata_df)
            plot_classification_examples(
                X_test, y_test, X_test_metadata_df, best_model_overall_info, output_base_dir
            )
        else:
            logging.warning(
                "No successful runs found to determine the best model for example plots."
            )

    else:
        logging.info("No experiment results to save.")

    logging.info("--- CPUE Classifier Workflow Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CPUE Classifier Workflow. "
        "Provide --num_kernels, --max_dilations, --n_features to run a single configuration. "
        "If none are provided, a predefined grid search will be executed."
    )
    # Make arguments optional for grid search, but still allow single run
    parser.add_argument(
        "--num_kernels", type=int, help="Number of kernels for MultiRocket."
    )
    parser.add_argument("--max_dilations", type=int, help="Max dilations per kernel.")
    parser.add_argument("--n_features", type=int, help="Number of features per kernel.")
    parser.add_argument(
        "--C",
        type=float,
        help="C parameter for Logistic Regression. Only used if MultiRocket args are also provided.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)