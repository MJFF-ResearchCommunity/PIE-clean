import glob
import logging
import os
import pandas as pd
import numpy as np

from pie_clean.utils import aggregate_by_patno_eventid, general_deduplicate_suffixed_columns

logger = logging.getLogger(f"PIE.{__name__}")

FILE_PREFIXES = [
    "Age_at_visit",
    "Demographics",
    "Family_History",
    "iu_genetic_consensus",
    "Participant_Status",
    "PPMI_PD_Variants",
    "PPMI_Project_9001",
    "Socio-Economics",
    "Subject_Cohort_History"
]


def load_ppmi_subject_characteristics(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files for subject characteristics.
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
                logger.debug(f"Loading subject characteristics file: {csv_file_path}")
                df_temp = pd.read_csv(csv_file_path, low_memory=False)
                found_any_file = True
            except Exception as e:
                logger.error(f"Could not read file '{csv_file_path}': {e}")
                continue

            if "PATNO" not in df_temp.columns:
                logger.warning(f"File {csv_file_path} is missing PATNO column, skipping.")
                continue

            # Standardize PATNO to string early
            df_temp['PATNO'] = df_temp['PATNO'].astype(str)


            if df_merged is None:
                df_merged = df_temp
            else:
                # Ensure df_merged PATNO is string
                if 'PATNO' in df_merged.columns:
                     df_merged['PATNO'] = df_merged['PATNO'].astype(str)

                
                merge_keys = ["PATNO"]
                # Merge on EVENT_ID only if both frames have it
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
                    logger.error(f"Error merging {os.path.basename(csv_file_path)} into df_merged: {e}")
                    logger.error(f"df_merged columns: {df_merged.columns.tolist()}")
                    logger.error(f"df_temp columns: {df_temp.columns.tolist()}")
                    logger.error(f"Merge keys: {merge_keys}")
                    continue # Skip this file if merge fails

    if not found_any_file or df_merged is None or df_merged.empty:
        logger.warning("No matching subject characteristics CSV files were successfully loaded or merged. Returning empty DataFrame.")
        return pd.DataFrame()

    # 1. Resolve _x, _y suffixed columns resulting from merges
    # 2. Ensure (PATNO, EVENT_ID) uniqueness by aggregating rows
    # This step must happen after all files are merged and _x/_y columns are resolved.
    if "EVENT_ID" in df_merged.columns:
        logger.debug("Aggregating rows to ensure unique (PATNO, EVENT_ID) pairs for Subject Characteristics...")
        df_merged = aggregate_by_patno_eventid(df_merged, "Subject Characteristics")
    else:
        logger.warning("EVENT_ID column not found in the final merged subject characteristics DataFrame. "
                       "Ensuring PATNO uniqueness only if duplicates exist.")
        if "PATNO" in df_merged.columns and df_merged.duplicated(subset=["PATNO"]).any():
             # For PATNO-only aggregation, we'll use the same logic
             logger.info(
                "Subject Characteristics: Consolidating rows with duplicate PATNO "
                "by combining unique non-null values for other columns."
             )
             # Temporarily rename PATNO for the groupby function if EVENT_ID is missing
             # This is a bit of a hack to reuse the same aggregate_by_patno_eventid logic
             # Or better, adapt aggregate_by_patno_eventid to handle single key
             
             # Simplified aggregation for PATNO only if EVENT_ID is missing
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
                 df_merged['PATNO'] = df_merged['PATNO'].astype(str)
                 df_merged = df_merged.groupby("PATNO", as_index=False).agg(agg_dict_patno)
             else: # Only PATNO column exists
                 df_merged = df_merged.drop_duplicates(subset=["PATNO"], keep='first')
        
    logger.info(f"Final loaded subject characteristics shape: {df_merged.shape}")
    return df_merged


# def main():
#     """
#     Example usage of load_ppmi_subject_characteristics:
#     If some CSVs have only PATNO (no EVENT_ID),
#     those columns will be replicated across all event rows for that PATNO.
#     """
#     path_to_subject_characteristics = "./PPMI/_Subject_Characteristics"
#     df_subjects = load_ppmi_subject_characteristics(path_to_subject_characteristics)
#     logger.info(df_subjects.head(25))  # Show first rows to see merge results
#     df_subjects.to_csv("subject_characteristics.csv", index=False)

# if __name__ == "__main__":
#     main()
