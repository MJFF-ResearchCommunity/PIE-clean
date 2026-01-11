import logging
import os
import pandas as pd
import numpy as np

from pie_clean.utils import load_all_files, merge_data_dict
from pie_clean.constants import *

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
    if not os.path.exists(folder_path):
        logger.warning(f"Directory not found: {folder_path}")
        return pd.DataFrame()

    df_dict = load_all_files(folder_path, FILE_PREFIXES, SUBJECT_CHARACTERISTICS)
    df_merged = merge_data_dict(SUBJECT_CHARACTERISTICS, df_dict)

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
