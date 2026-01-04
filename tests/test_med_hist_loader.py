import sys
import shutil
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie_clean.med_hist_loader import (
    load_ppmi_medical_history,
    MEDICAL_HISTORY_PREFIXES
)

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"
SUB_DIR = "Medical_History"

load_msg = f"Loading medical_history file: {DATA_DIR}/{SUB_DIR}"

def test_load_ppmi_medical_history(caplog):
    # Returns dict of dfs, not a merged df
    df_dict = load_ppmi_medical_history(DATA_DIR)

    # We expect to see certain files, which are handled in specific ways
    # The logger output clarifies how they were handled
    # First, just check that we're looking for everything
    for filename in MEDICAL_HISTORY_PREFIXES:
        assert filename in caplog.text

    # Now pick out some specifics
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
    assert "No matching medical_history" in record.message
    assert len(df_dict) == 0, "Expected empty dict, contains {len(df_dict)} items"

