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
        elif "Neuro_QoL" in record.message:
            # Load only the Motor versions of Neuro_QoL, not the Non-motor versions
            assert f"{load_msg}/Neuro_QoL__Lower_Extremity" in record.message or\
                   f"Neuro_QoL__Upper_Extremity" in record.message
            # We've mocked up Lower_Extremity, so it must be actually loaded
            assert f"{load_msg}/Neuro_QoL__Lower_Extremity" in caplog.text
            assert "Upper_Extremity" in caplog.text # Not mocked, so not actually loaded
            # And we must not see the Non-motor file
            assert "Neuro_QoL__Cognition_Function" not in caplog.text,\
                   "Motor loader is incorrectly loading Non-motor Neuro_QoL"

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

    # Ensure columns have been deduplicated correctly. NUPSOURC appears in all UPDRS tables
    assert "NUPSOURC" in df.columns.tolist()
    assert "NUPSOURC_x" not in df.columns.tolist()
    assert "NUPSOURC_y" not in df.columns.tolist()
    # All ones get merged to a single one, with the original type
    assert df[(df["PATNO"]=="9999")&(df["EVENT_ID"]=="BL")].iloc[0,:]["NUPSOURC"] == 1
    # Different values get merged into a pipe-separated string
    assert df[(df["PATNO"]=="9999")&(df["EVENT_ID"]=="V04")].iloc[0,:]["NUPSOURC"] == "1|2"
    assert df[(df["PATNO"]=="9999")&(df["EVENT_ID"]=="V06")].iloc[0,:]["NUPSOURC"] == "1|2|3"

def test_empty_dir(caplog, tmp_path):
    df = load_ppmi_motor_assessments(tmp_path)

    record = caplog.records[-1] # Last log message
    assert record.levelname == "WARNING"
    assert "No matching motor assessment CSV files" in record.message
    assert df.empty

def test_missing_patno(caplog, tmp_path):
    # Set up the empty PATNO file among others
    testfile = "MDS-UPDRS_Part_I_Patient_Questionnaire"
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    file = tmp_path / SUB_DIR / f"{testfile}_21Test2025.csv"
    tmp = pd.read_csv(file)
    tmp = tmp.drop(columns="PATNO")
    tmp.to_csv(file, index=False)

    df = load_ppmi_motor_assessments(tmp_path)

    records = [r for r in caplog.records if testfile in r.message]
    assert len(records) == 2, f"Should be 2 log records for {testfile}: found {len(records)}"

    assert "Loading motor assessment file" in records[0].message # normal loading msg
    assert records[1].levelname == "WARNING"
    assert "missing PATNO column, skipping" in records[1].message

    # And the output shouldn't contain NP1SLPD
    assert "NP1SLPD" not in df.columns.tolist()
