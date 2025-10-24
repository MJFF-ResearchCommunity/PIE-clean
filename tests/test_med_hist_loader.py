import sys
import shutil
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie_clean.med_hist_loader import load_ppmi_medical_history, sanitize_suffixes_in_df

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"
SUB_DIR = "Medical_History"

load_msg = f"Loading medical history file: {DATA_DIR}/{SUB_DIR}"

def test_load_ppmi_medical_history(caplog):
    # Returns dict of dfs, not a merged df
    df_dict = load_ppmi_medical_history(DATA_DIR)

    # We expect to see certain files, which are handled in specific ways
    # The logger output clarifies how they were handled
    assert "General_Physical_Exam" in caplog.text
    assert "LEDD_Concomitant_Medication" in caplog.text
    assert "Vital_Signs" in caplog.text

    for record in caplog.records:
        if "General_Physical_Exam" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/General_Physical_Exam" in record.message
        if "LEDD_Concomitant_Medication" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/LEDD_Concomitant_Medication" in record.message
        if "Vital_Signs" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/Vital_Signs" in record.message

    # We expect to see every table in the output
    assert "General_Physical_Exam" in df_dict
    assert "PECAT" in df_dict["General_Physical_Exam"].columns
    assert "LEDD_Concomitant_Medication" in df_dict
    assert "LEDTRT" in df_dict["LEDD_Concomitant_Medication"].columns
    assert "Vital_Signs" in df_dict
    assert "SYSSUP" in df_dict["Vital_Signs"].columns

def test_empty_dir(caplog, tmp_path):
    df_dict = load_ppmi_medical_history(tmp_path)

    record = caplog.records[-1] # Last log message
    assert record.levelname == "WARNING"
    assert "No matching medical history" in record.message
    assert len(df_dict) == 0, "Expected empty dict, contains {len(df_dict)} items"

def test_sanitize_suffixes(caplog):
    # Set up a table with "_x" and "_y" columns
    tmp = pd.read_csv(f"{DATA_DIR}/{SUB_DIR}/General_Physical_Exam_21Test2025.csv")
    c1, c2 = "TEST", "TEST2"
    tmp[f"{c1}_x"] = [0] * tmp.shape[0]
    tmp[f"{c1}_y"] = [1] * tmp.shape[0]
    tmp[f"{c2}_y"] = [2] * tmp.shape[0]

    sanitize_suffixes_in_df(tmp) # operates in place
    assert f"{c1}_x" not in tmp.columns.tolist()
    assert f"{c1}_y" not in tmp.columns.tolist()
    assert f"{c2}_y" not in tmp.columns.tolist()
    assert f"{c1}_col" in tmp.columns
    assert f"{c1}_col1" in tmp.columns
    assert tmp[f"{c1}_col"].iloc[0] == 0 # First in iloc order gets "_col"
    assert tmp[f"{c1}_col1"].iloc[0] == 1 # Second gets "_col1"
    assert f"{c2}_col" in tmp.columns # Third is a different name, so "_col"

