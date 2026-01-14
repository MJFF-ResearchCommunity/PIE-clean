import glob
import os
import pandas as pd
import numpy as np
import logging

from pie_clean.utils import load_all_files, merge_data_dict
from pie_clean.constants import *

logger = logging.getLogger(f"PIE.{__name__}")

# List of file prefixes to search for in the Non-motor_Assessments folder
FILE_PREFIXES = [
    "Benton_Judgement",
    "Clock_Drawing",
    "Cognitive_Categorization",
    "Cognitive_Change",
    "Epworth_Sleepiness_Scale",
    "Geriatric_Depression_Scale",
    "Hopkins_Verbal_Learning_Test",
    "IDEA_Cognitive_Screen",
    "Letter_-_Number_Sequencing",
    "Lexical_Fluency",
    "Modified_Boston_Naming_Test",
    "Modified_Semantic_Fluency",
    "Montreal_Cognitive_Assessment",
    "Neuro_QoL__Cognition", # Note: Neuro_QoL also appears in Motor. Need to be specific
    "Neuro_QoL__Communication",
    "PDAQ-27",
    "QUIP-Current-Short",
    "REM_Sleep_Behavior_Disorder_Questionnaire",
    "SCOPA-AUT",
    "State-Trait_Anxiety_Inventory",
    "Symbol_Digit_Modalities",
    "Trail_Making",
    "University_of_Pennsylvania_Smell_Identification"
]


def load_ppmi_non_motor_assessments(folder_path: str) -> pd.DataFrame:
    """
    Loads and merges CSV files for non-motor assessments.
    Ensures unique (PATNO, EVENT_ID) rows in the output by merging information.
    """
    if not os.path.exists(folder_path):
        logger.warning(f"Directory not found: {folder_path}")
        return pd.DataFrame()

    df_dict = load_all_files(folder_path, FILE_PREFIXES, NON_MOTOR_ASSESSMENTS)
    df_merged = merge_data_dict(NON_MOTOR_ASSESSMENTS, df_dict)

    return df_merged


# def main():
#     """
#     Example usage of load_ppmi_non_motor_assessments:
#     Loads and merges all non-motor assessment files from the PPMI/Non-motor_Assessments folder.
#     """
#     path_to_non_motor_assessments = "./PPMI/Non-motor_Assessments"
    
#     # Print all CSV files in the PPMI directory to help debug
#     print("[INFO] Listing all CSV files in the Non-motor_Assessments directory:")
#     if os.path.exists(path_to_non_motor_assessments):
#         for root, dirs, files in os.walk(path_to_non_motor_assessments):
#             for file in files:
#                 if file.lower().endswith('.csv'):
#                     print(f"  - {os.path.join(root, file)}")
#     else:
#         print(f"[WARNING] Directory not found: {path_to_non_motor_assessments}")
    
#     df_non_motor = load_ppmi_non_motor_assessments(path_to_non_motor_assessments)
#     print(df_non_motor.head(25))  # Show first rows to see merge results
#     df_non_motor.to_csv("ppmi_non_motor_assessments.csv", index=False)

# if __name__ == "__main__":
#     main()
