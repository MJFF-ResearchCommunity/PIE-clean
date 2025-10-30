import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(f"PIE.{__name__}")

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
