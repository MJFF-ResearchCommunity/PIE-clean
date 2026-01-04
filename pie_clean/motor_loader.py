import glob
import os
import pandas as pd
import numpy as np
import logging

from pie_clean.utils import load_all_files
from pie_clean.constants import *

logger = logging.getLogger(f"PIE.{__name__}")

# List of file prefixes to search for in the Motor___MDS-UPDRS folder
FILE_PREFIXES = [
    "Gait_Data___Arm_swing",
    "Gait_Substudy_Gait_Mobility_Assessment",
    "MDS-UPDRS_Part_I", # Picks up Parts I, II, III, IV, and _Patient versions
    "Modified_Schwab",
    "Neuro_QoL__Lower_Extremity", # Note: Neuro_QoL also appears in Non-motor. Need to be specific
    "Neuro_QoL__Upper_Extremity",
    "Participant_Motor_Function"
]

def load_ppmi_motor_assessments(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files for motor assessments.
    Ensures unique (PATNO, EVENT_ID) rows in the output by merging information.
    """
    if not os.path.exists(folder_path):
        logger.warning(f"Directory not found: {folder_path}")
        return pd.DataFrame()

    df_merged = load_all_files(folder_path, FILE_PREFIXES, MOTOR_ASSESSMENTS, merge=True)

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
