import logging
import glob
import os
import pandas as pd
import numpy as np

from pie_clean.utils import load_all_files
from pie_clean.constants import *

logger = logging.getLogger(f"PIE.{__name__}")

# Many medical history files cannot be merged, because data is recorded multiple times per visit,
# or on a timeline that doesn't correspond to the visits. Examples of the first are Adverse Event
# logs, where at the visit, multiple adverse events over the past period can be recorded. An
# example of the second is Concomitant Meds, which have start and end dates based on when the
# medication was taken. As a result, medical history tables should be kept separate.

MEDICAL_HISTORY_PREFIXES = [
    "Adverse_Event",
    "AV-133_Prodromal",
    "C05-05_PET_Imaging_Substudy",
    "Clinical_Diagnosis",
    "Clinical_Global_Impression",
    "Concomitant_Medication",
    "Determination_of_Freezing_and_Falls",
    "DPA-714_PET_Imaging_Substudy_Adverse_Event",
    "Early_Imaging",
    "Features_of_Parkinsonism",
    "Features_of_REM_Behavior_Disorder",
    "Gait_Substudy_Adverse_Event",
    "General_Physical_Exam",
    "Initiation_of_Dopaminergic_Therapy",
    "LEDD_Concomitant_Medication",
    "Medical_Conditions",
    "Neurological_Exam",
    "Other_Clinical_Features",
    "Participant_Global_Impression",
    "PD_Diagnosis_History",
    "Pregnancy_Test",
    "Primary_Clincial_Diagnosis",
    "Procedure_for_PD_Log",
    "Report_of_Pregnancy",
    "SVA2_PET_Imaging_Substudy",
    "Tau_Substudy",
    "Vital_Signs"
]


def load_ppmi_medical_history(folder_path: str) -> dict:
    """
    1) Lists all CSV files in 'folder_path' that start with any MEDICAL_HISTORY_PREFIX.
    2) For each CSV, read into df_temp.
         - sanitize_suffixes_in_df(df_temp), in case it is merged in the future
         - Store in a dict of tables, with the prefix as the key
    3) Return the dict or empty if no files found.
    """
    if not os.path.exists(folder_path):
        logger.warning(f"Directory not found: {folder_path}")
        return {}

    df_dict = load_all_files(folder_path, MEDICAL_HISTORY_PREFIXES, MEDICAL_HISTORY, merge=False)

    return df_dict


# def main():
#     path_to_med_history = "./PPMI/Medical_History"
#     med_history = load_ppmi_medical_history(path_to_med_history)
#     logger.info(sorted(list(med_history.keys())))
# if __name__ == "__main__":
#     main()
