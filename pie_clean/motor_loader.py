import glob
import os
import pandas as pd
import numpy as np
import logging

from pie_clean.utils import (
    aggregate_by_patno_eventid, general_deduplicate_suffixed_columns
)

logger = logging.getLogger(f"PIE.{__name__}")

# List of file prefixes to search for in the Motor___MDS-UPDRS folder
FILE_PREFIXES = [
    "Gait_Data___Arm_swing",
    "Gait_Substudy_Gait_Mobility_Assessment",
    "MDS-UPDRS_Part_I", # Picks up Parts I, II, III, IV, and _Patient versions
    "Modified_Schwab",
    "Neuro_QoL", # Note: Neuro_QoL also appears in Non-motor. Prefixes will ensure correct loading.
    "Participant_Motor_Function"
]

def _sanitize_suffixes_in_df(df: pd.DataFrame) -> None:
    """
    Rename columns in df if they already end with '_x' or '_y',
    so that Pandas won't clash when merging with suffixes=('_x', '_y').
    Example: 'COL_x' -> 'COL_x_orig'.
    """
    rename_map = {}
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y"):
            base = col[:-2]
            new_col_candidate = f"{base}_{col[-1]}_orig" # e.g. SOME_COL_x_orig
            # Ensure new_col_candidate is unique
            count = 0
            new_col = new_col_candidate
            while new_col in df.columns or new_col in rename_map.values():
                count += 1
                new_col = f"{new_col_candidate}{count}"
            rename_map[col] = new_col
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        logger.debug(f"Sanitized existing suffixed columns: {rename_map}")

def load_ppmi_motor_assessments(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files for motor assessments.
    Ensures unique (PATNO, EVENT_ID) rows in the output by merging information.
    """
    df_merged = None
    found_any_file = False

    all_csv_files = list(glob.iglob(os.path.join(folder_path, "**/*.csv"), recursive=True))

    for prefix in FILE_PREFIXES:
        matching_files = [f for f in all_csv_files if os.path.basename(f).startswith(prefix)]
        if not matching_files:
            logger.debug(f"No CSV file found for prefix: {prefix} in {folder_path}")
            continue

        for csv_file_path in matching_files:
            try:
                logger.debug(f"Loading motor assessment file: {csv_file_path}")
                df_temp = pd.read_csv(csv_file_path, low_memory=False)
                found_any_file = True
            except Exception as e:
                logger.error(f"Could not read file '{csv_file_path}': {e}")
                continue

            if "PATNO" not in df_temp.columns:
                logger.warning(f"File {csv_file_path} is missing PATNO column, skipping.")
                continue

            df_temp['PATNO'] = df_temp['PATNO'].astype(str)

            if df_merged is None:
                df_merged = df_temp
            else:
                if 'PATNO' in df_merged.columns:
                     df_merged['PATNO'] = df_merged['PATNO'].astype(str)

                merge_keys = ["PATNO"]
                if "EVENT_ID" in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    merge_keys.append("EVENT_ID")
                elif "EVENT_ID" in df_merged.columns and "EVENT_ID" not in df_temp.columns:
                    logger.debug(f"Merging {os.path.basename(csv_file_path)} on PATNO only (it lacks EVENT_ID).")
                elif "EVENT_ID" not in df_merged.columns and "EVENT_ID" in df_temp.columns:
                    logger.debug(f"Merging {os.path.basename(csv_file_path)} on PATNO only (df_merged lacks EVENT_ID).")

                try:
                    df_merged = pd.merge(df_merged, df_temp, on=merge_keys, how="outer", suffixes=('_x', '_y'))
                    df_merged = general_deduplicate_suffixed_columns(df_merged)
                except Exception as e:
                    logger.error(f"Error merging {os.path.basename(csv_file_path)} into motor df_merged: {e}")
                    logger.error(f"df_merged columns: {df_merged.columns.tolist()}")
                    logger.error(f"df_temp columns: {df_temp.columns.tolist()}")
                    logger.error(f"Merge keys: {merge_keys}")
                    continue

    if not found_any_file or df_merged is None or df_merged.empty:
        logger.warning("No matching motor assessment CSV files were successfully loaded or merged. Returning empty DataFrame.")
        return pd.DataFrame()

    if "EVENT_ID" in df_merged.columns:
        logger.debug("Motor assessments: Aggregating rows to ensure unique (PATNO, EVENT_ID) pairs...")
        df_merged = aggregate_by_patno_eventid(df_merged, "Motor assessments")
    else:
        logger.warning("EVENT_ID column not found in the final merged motor assessments DataFrame. "
                       "Ensuring PATNO uniqueness only if duplicates exist.")
        if "PATNO" in df_merged.columns and df_merged.duplicated(subset=["PATNO"]).any():
             logger.info(
                "Motor Assessments: Consolidating rows with duplicate PATNO "
                "by combining unique non-null values for other columns."
             )
             def combine_patno_only_series(series):
                unique_non_null_strs = series.dropna().astype(str).unique()
                if len(unique_non_null_strs) == 0: return np.nan
                if len(unique_non_null_strs) == 1:
                    original_non_null_values = series.dropna()
                    if original_non_null_values.nunique() == 1:
                        return original_non_null_values.iloc[0]
                    return unique_non_null_strs[0]
                return "|".join(sorted(unique_non_null_strs))

             agg_cols_patno = [col for col in df_merged.columns if col != "PATNO"]
             if agg_cols_patno:
                 agg_dict_patno = {col: combine_patno_only_series for col in agg_cols_patno}
                 df_merged['PATNO'] = df_merged['PATNO'].astype(str) # Ensure PATNO is string
                 df_merged = df_merged.groupby("PATNO", as_index=False).agg(agg_dict_patno)
             else: 
                 df_merged = df_merged.drop_duplicates(subset=["PATNO"], keep='first')

    logger.info(f"Final loaded motor assessments shape: {df_merged.shape}")
    return df_merged


# def main():
#     """
#     Example usage of load_ppmi_motor_assessments:
#     Loads and merges all motor assessment files from the PPMI/Motor___MDS-UPDRS folder.
#     """
#     path_to_motor_assessments = "./PPMI/Motor___MDS-UPDRS"
    
#     # Print all CSV files in the PPMI directory to help debug
#     print("[INFO] Listing all CSV files in PPMI directory:")
#     for root, dirs, files in os.walk("./PPMI"):
#         for file in files:
#             if file.lower().endswith('.csv'):
#                 print(f"  - {os.path.join(root, file)}")
    
#     df_motor = load_ppmi_motor_assessments(path_to_motor_assessments)
#     print(df_motor.head(25))  # Show first rows to see merge results
#     df_motor.to_csv("ppmi_motor_assessments.csv", index=False)

# if __name__ == "__main__":
#     main()
