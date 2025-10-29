import sys
import shutil
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie_clean.motor_loader import (
    load_ppmi_motor_assessments, FILE_PREFIXES
)

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"
SUB_DIR = "Motor___MDS-UPDRS" 

load_msg = f"Loading motor assessment file: {DATA_DIR}/{SUB_DIR}"

def test_load_ppmi_motor_assessments(caplog):
    df = load_ppmi_motor_assessments(DATA_DIR)

    # We expect to see certain files, which are handled in specific ways
    # The logger output clarifies how they were handled
    # First, just check that we're looking for everything
    for filename in FILE_PREFIXES:
        assert filename in caplog.text

    # Now pick out some specifics
    for record in caplog.records:
        if "Part_I_Patient" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/MDS-UPDRS_Part_I_Patient" in record.message
        elif "Part_I_" in record.message: # Already eliminated Patient versions
            # One and only one message, for loading
            assert f"{load_msg}/MDS-UPDRS_Part_I_" in record.message
        elif "Part_II_Patient" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/MDS-UPDRS_Part_II_Patient" in record.message

    # We expect to see something from every table merged together in the output
    assert "PATNO" in df.columns
    assert "EVENT_ID" in df.columns
    assert "NP1COG" in df.columns
    assert "NP1SLPN" in df.columns
    assert "NP2SPCH" in df.columns

    # Ensure that EVENT_IDs across tables have been properly merged
    pat = df[df["PATNO"]=="9999"]
    assert "BL" in pat["EVENT_ID"].tolist()
    assert "V04" in pat["EVENT_ID"].tolist()
    assert pat[pat["EVENT_ID"]=="BL"].shape[0] == 1 # Only one row
    assert pat[pat["EVENT_ID"]=="V04"].shape[0] == 1 # Also only one row
    assert pat[pat["EVENT_ID"]=="BL"]["NP1SLPD"].iloc[0] == 0
    assert pat[pat["EVENT_ID"]=="V04"]["NP1SLPD"].iloc[0] == 1
    assert pat[pat["EVENT_ID"]=="BL"]["NP1DPRS"].iloc[0] == 0
    assert pat[pat["EVENT_ID"]=="V04"]["NP1DPRS"].iloc[0] == 1
    assert pat[pat["EVENT_ID"]=="BL"]["NP2EAT"].iloc[0] == 0
    assert pat[pat["EVENT_ID"]=="V04"]["NP2EAT"].iloc[0] == 1

def test_empty_dir(caplog, tmp_path):
    df = load_ppmi_motor_assessments(tmp_path)

    record = caplog.records[-1] # Last log message
    assert record.levelname == "WARNING"
    assert "No matching motor assessment CSV files" in record.message
    assert df.empty
