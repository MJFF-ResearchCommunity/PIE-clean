from __future__ import annotations
import glob
import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(f"PIE.{__name__}")

def aggregate_by_patno_eventid(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """
    Ensures (PATNO, EVENT_ID) pairs are unique by grouping and aggregating.
    For non-grouping columns, it combines unique non-null string values with a pipe.
    If only one unique non-null value exists, it's used directly (attempting to keep original type).
    """
    if df.empty:
        return df

    group_cols = ["PATNO", "EVENT_ID"]
    if not all(gc in df.columns for gc in group_cols):
        logger.warning(f"{modality}: Cannot aggregate by {group_cols} as one or more are missing. Returning original DataFrame.")
        return df

    df_copy = df.copy()
    if 'PATNO' in df_copy.columns:
        df_copy['PATNO'] = df_copy['PATNO'].astype(str)

    if not df_copy.duplicated(subset=group_cols).any():
        return df_copy

    logger.info(
        f"{modality}: Consolidating rows with duplicate (PATNO, EVENT_ID) pairs. "
        "Non-null values for other columns will be pipe-separated if different."
    )

    agg_cols = [col for col in df.columns if col not in group_cols]
    if not agg_cols:
        return df_copy.drop_duplicates(subset=group_cols, keep='first')

    df_indexed = df_copy.set_index(group_cols)

    grouped = df_indexed.groupby(level=group_cols)
    nunique_df = grouped[agg_cols].nunique()
    result_df = grouped[agg_cols].first()

    pipe_separated_stats = {}

    for col in agg_cols:
        multi_value_groups_mask = nunique_df[col] > 1
        if not multi_value_groups_mask.any():
            continue

        num_affected_groups = multi_value_groups_mask.sum()
        if num_affected_groups > 0:
            pipe_separated_stats[col] = num_affected_groups

        multi_value_group_indices = nunique_df.index[multi_value_groups_mask]

        rows_for_col_agg_mask = df_indexed.index.isin(multi_value_group_indices)
        df_subset_for_col = df_indexed.loc[rows_for_col_agg_mask, [col]]

        if df_subset_for_col.empty:
            continue

        def string_agg_slow(series: pd.Series) -> str:
            unique_strings = series.dropna().astype(str).unique()
            return "|".join(sorted(unique_strings))

        slow_agg_results = df_subset_for_col.groupby(level=group_cols)[col].agg(string_agg_slow)

        # If the target column is not already an object/string type, cast it.
        # This prevents FutureWarning about incompatible dtypes.
        if result_df[col].dtype != 'object' and not pd.api.types.is_string_dtype(result_df[col].dtype):
            result_df[col] = result_df[col].astype(object)

        result_df.loc[slow_agg_results.index, col] = slow_agg_results

    df_aggregated = result_df.reset_index()

    if pipe_separated_stats:
        logger.info(f"Summary of pipe-separated columns for {modality}:")
        sorted_stats = sorted(pipe_separated_stats.items(), key=lambda item: item[1], reverse=True)

        for i, (col, count) in enumerate(sorted_stats):
            logger.info(f"  - Column '{col}': {count} groups had multiple values.")
            if i < 3: # Log examples for the top 3
                first_offending_group_index = nunique_df.index[nunique_df[col] > 1][0]
                conflicting_values = df_indexed.loc[first_offending_group_index, col].dropna().astype(str).unique()
                logger.info(f"    - Example for group {first_offending_group_index}: values were {list(conflicting_values)}")

    ordered_cols = group_cols + [col for col in df.columns if col in df_aggregated.columns and col not in group_cols]
    final_ordered_cols = [col for col in ordered_cols if col in df_aggregated.columns]

    return df_aggregated[final_ordered_cols]

def general_deduplicate_suffixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies all columns with '_x' and '_y' suffixes, then merges them
    into a base column name.
    - If only one of col_x or col_y exists, it's renamed to base_col.
    - If both exist, their values are combined:
        - If one is NaN, the other is used.
        - If both are non-NaN and equal, one is used.
        - If both are non-NaN and different, they are pipe-separated.
    """
    if df.empty:
        return df

    cols_to_process = set()
    for col_name in df.columns:
        if col_name.endswith('_x'):
            cols_to_process.add(col_name[:-2])
        elif col_name.endswith('_y'):
            cols_to_process.add(col_name[:-2])

    if not cols_to_process:
        return df

    logger.debug(f"Deduplicating suffixed columns for bases: {cols_to_process}")

    for base_col_name in list(cols_to_process): # Iterate over a copy
        col_x = f"{base_col_name}_x"
        col_y = f"{base_col_name}_y"

        has_x = col_x in df.columns
        has_y = col_y in df.columns

        if has_x and has_y:
            logger.debug(f"Combining {col_x} and {col_y} into {base_col_name}")
            # Ensure base_col_name doesn't overwrite an existing non-suffixed column
            # that wasn't part of this _x/_y pair (should be rare if sanitization worked)
            if base_col_name in df.columns and base_col_name != col_x and base_col_name != col_y:
                 logger.warning(f"Base column {base_col_name} already exists. Combining _x/_y may overwrite it.")

            def combine_values(row):
                v1 = row[col_x]
                v2 = row[col_y]
                is_empty_1 = pd.isna(v1) or str(v1).strip() == ""
                is_empty_2 = pd.isna(v2) or str(v2).strip() == ""

                if is_empty_1 and is_empty_2: return np.nan
                elif is_empty_1: return v2
                elif is_empty_2: return v1
                else: # Both are non-empty
                    # Convert to string for comparison to handle mixed types robustly
                    s_v1, s_v2 = str(v1), str(v2)
                    if s_v1 == s_v2:
                        return v1 # Return original type if possible
                    else:
                        # Attempt to convert to a common numeric type if possible before string concatenation
                        try:
                            f_v1 = float(v1)
                            f_v2 = float(v2)
                            if np.isclose(f_v1, f_v2): return v1
                        except (ValueError, TypeError):
                            pass # Not both convertible to float, or one is string etc.
                        return f"{s_v1}|{s_v2}"

            df[base_col_name] = df.apply(combine_values, axis=1)
            df.drop(columns=[col_x, col_y], inplace=True)
        elif has_x: # Only _x exists
            logger.debug(f"Renaming {col_x} to {base_col_name}")
            df.rename(columns={col_x: base_col_name}, inplace=True)
        elif has_y: # Only _y exists
            logger.debug(f"Renaming {col_y} to {base_col_name}")
            df.rename(columns={col_y: base_col_name}, inplace=True)
    return df

def sanitize_suffixes_in_df(df: pd.DataFrame) -> None:
    """
    Rename columns in df if they already end with '_x' or '_y'.
    For example:
      'SUB_EVENT_ID_x' -> 'SUB_EVENT_ID_x_col'
      'SUB_EVENT_ID_y' -> 'SUB_EVENT_ID_y_col'
    If 'SUB_EVENT_ID_x_col' also exists, keep incrementing => 'SUB_EVENT_ID_x_col1', etc.
    """
    rename_map = {}
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y"):
            base = col  # e.g. 'SUB_EVENT_ID_x'
            # Chop off the last two chars ('_x' or '_y') and append '_col'
            new_base = base[:-2] + "_col"  # e.g. 'SUB_EVENT_ID_col'
            new_col = new_base

            # If that new_col also exists, keep incrementing
            count = 1
            while new_col in df.columns or new_col in rename_map.values():
                new_col = f"{new_base}{count}"
                count += 1

            rename_map[col] = new_col

    if rename_map:
        df.rename(columns=rename_map, inplace=True)


def find_all_files(modality: str, folder_path: str, file_prefixes: list[str]):
    all_csv_files = list(glob.iglob(os.path.join(folder_path, "**/*.csv"), recursive=True))

    all_files = []
    for prefix in file_prefixes:
        matching_files = [f for f in all_csv_files if os.path.basename(f).startswith(prefix)]
        if not matching_files:
            logger.debug(f"{modality}: No CSV file found for prefix: {prefix} in {folder_path}")
            continue
        all_files.extend(matching_files)
    return all_files

def load_single_file(modality: str, full_file_path: str):
    logger.debug(f"Loading {modality} file: {full_file_path}")
    try:
        df = pd.read_csv(full_file_path, low_memory=False)
    except Exception as e:
        logger.error(f"{modality}: Could not read file '{full_file_path}': {e}")
        return pd.DataFrame()

    if "PATNO" not in df.columns:
        logger.warning(f"{modality} file {full_file_path} is missing PATNO column, skipping.")
        return pd.DataFrame()

    df['PATNO'] = df['PATNO'].astype(str) # Standardize PATNO immediately
    # Remove "PPMI-" prefix from PATNO if it exists
    df["PATNO"] = df["PATNO"].apply(
        lambda x: x[5:] if isinstance(x, str) and x.startswith("PPMI-") else x
    )
    if "CLINICAL_EVENT" in df.columns:
        df = df.rename(columns={"CLINICAL_EVENT": "EVENT_ID"})
    sanitize_suffixes_in_df(df) # Rename columns ending _x or _y
    return df

def load_all_files(
    folder_path: str,
    file_prefixes: list[str],
    modality: str,
    merge: bool = False
):
    data_dict = {}

    matching_files = find_all_files(modality, folder_path, file_prefixes)

    for csv_file_path in sorted(matching_files):
        df_temp = load_single_file(modality, csv_file_path)
        if df_temp.empty:
            continue # Error was described in the function

        # Create a good key for storage: filename minus "[date].csv"
        filename = os.path.basename(csv_file_path)
        filename = "_".join(filename.split("_")[:-1])
        if filename in data_dict:
            logger.warning(f"{modality}: unexpected duplicate filename {filename} found: dropping one")
        data_dict[filename] = df_temp

    if len(data_dict) == 0:
        logger.warning(f"No matching {modality} CSV files were successfully loaded.")
    else:
        logger.info(f"{modality}: Returning dict of {len(data_dict)} files")
    return data_dict

def merge_data_dict(modality: str, data_dict: dict[str, pd.DataFrame]):
    if len(data_dict) == 0:
        return pd.DataFrame() # Nothing to merge

    df_merged = None
    for i, key in enumerate(data_dict):
        if i == 0:
            df_merged = data_dict[key]
            continue

        merge_keys = ["PATNO"]
        df_temp = data_dict[key]
        if "EVENT_ID" in df_merged and "EVENT_ID" in df_temp:
            logger.debug(f"Merging {modality} {key} on PATNO and EVENT_ID")
            merge_keys.append("EVENT_ID")
        elif "EVENT_ID" in df_merged:
            logger.debug(f"Merging {modality} {key} on PATNO only (it lacks EVENT_ID)")
        else:
            logger.debug(f"Merging {modality} {key} on PATNO only (pre-existing df_merged lacks EVENT_ID)")

        try:
            df_merged = pd.merge(df_merged, df_temp, on=merge_keys, how="outer", suffixes=("_x", "_y"))
            df_merged = general_deduplicate_suffixed_columns(df_merged)
        except Exception as e:
            logger.error(f"Error merging {key} into df_merged: {e}")
            logger.error(f"df_merged columns: {df_merged.columns.tolist()}")
            logger.error(f"df_temp columns: {df_temp.columns.tolist()}")
            logger.error(f"Merge keys: {merge_keys}")
            continue # Skip this file if merge fails

    if df_merged.empty:
        logger.warning("No {modality} CSV files were successfully merged. Returning empty DataFrame.")
        return pd.DataFrame()

    # 1. Resolve _x, _y suffixed columns resulting from merges
    # 2. Ensure (PATNO, EVENT_ID) uniqueness by aggregating rows
    # This step must happen after all files are merged and _x/_y columns are resolved.
    if "EVENT_ID" in df_merged.columns:
        logger.debug(f"Aggregating rows to ensure unique (PATNO, EVENT_ID) pairs for {modality}...")
        df_merged = aggregate_by_patno_eventid(df_merged, modality)
    else:
        logger.warning(f"EVENT_ID column not found in the final merged {modality} DataFrame. "
                       "Ensuring PATNO uniqueness only if duplicates exist.")
        if "PATNO" in df_merged.columns and df_merged.duplicated(subset=["PATNO"]).any():
             # For PATNO-only aggregation, we'll use the same logic
             logger.info(
                f"{modality}: Consolidating rows with duplicate PATNO "
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

    logger.info(f"Final loaded {modality} shape: {df_merged.shape}")
    return df_merged

def pipe_separate_values(ser: pd.Series):
    """
    When multiple values exist for a given field for a unique <PATNO, EVENT_ID>
    pair, we handle it by including all values in a pipe-separated string. This
    function returns either the original value (if no combination is needed,
    because all duplicates are the same value), or a pipe-separated string. If
    the original values are strings, with duplicate values which differ only by
    capitalization, the values are upper-cased and pipe-separated, to give the
    minimal possible set.
    Examples (input -> return):
        pd.Series([1, 1, 1]) -> 1
        pd.Series([1, 2, 3]) -> "1|2|3"
        pd.Series([2, 1, 3]) -> "1|2|3"
        pd.Series(["YES", "YES", "YES"]) -> "YES"
        pd.Series(["yes", "no", "yes"]) -> "no|yes"
        pd.Series(["yes", "no", "YES"]) -> "NO|YES"
    """
    if ser.nunique() == 1: # No need for any combination
        return ser.iloc[0]
    if pd.api.types.is_object_dtype(ser) and \
            len(set(ser)) != len(set(ser.str.upper())):
        # We've got strings with duplicate values with inconsistent capitalization
        return "|".join(sorted(list(set(ser.dropna().str.upper())))) # upper case them all
    # Either not strings, or no inconsistent caps, so preserve format
    return "|".join([str(s) for s in sorted(list(set(ser.dropna())))])
