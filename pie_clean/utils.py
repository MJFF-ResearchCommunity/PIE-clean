import logging
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
