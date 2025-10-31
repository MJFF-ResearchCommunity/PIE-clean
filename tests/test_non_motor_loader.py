import sys
import shutil
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie_clean.non_motor_loader import (
    load_ppmi_non_motor_assessments, FILE_PREFIXES
)

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"
SUB_DIR = "Non-motor_Assessments" 

load_msg = f"Loading non-motor assessment file: {DATA_DIR}/{SUB_DIR}"

def test_load_ppmi_non_motor_assessments(caplog):
    df = load_ppmi_non_motor_assessments(DATA_DIR)

    # We expect to see certain files, which are handled in specific ways
    # The logger output clarifies how they were handled
    # First, just check that we're looking for everything
    for filename in FILE_PREFIXES:
        assert filename in caplog.text

    # Now pick out some specifics
    for record in caplog.records:
        if "MoCA" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/Montreal_Cognitive_Assessment" in record.message
        elif "Neuro_QoL" in record.message:
            # Load only the Non-motor versions of Neuro_QoL, not the Motor versions
            assert f"{load_msg}/Neuro_QoL__Cognition" in record.message or\
                   f"{load_msg}/Neuro_QoL__Communication" in record.message
            # We've mocked both, so they must be actually loaded
            assert f"{load_msg}/Neuro_QoL__Cognition" in caplog.text
            assert f"{load_msg}/Neuro_QoL__Communication" in caplog.text
            # And we must not see the Motor file which has been mocked
            assert "Neuro_QoL__Lower_Extremity" not in caplog.text,\
                   "Non-motor loader is incorrectly loading Motor Neuro_QoL"

    # We expect to see something from every table merged together in the output
    assert "PATNO" in df.columns
    assert "EVENT_ID" in df.columns
    assert "MCACLCKC" in df.columns
    assert "NQCOG04" in df.columns
    assert "NQCOG25R" in df.columns

    # Ensure that EVENT_IDs across tables have been properly merged
    pat = df[df["PATNO"]=="9999"]
    assert "SC" in pat["EVENT_ID"].tolist()
    assert "V04" in pat["EVENT_ID"].tolist()
    assert pat[pat["EVENT_ID"]=="SC"].shape[0] == 1 # Only one row
    assert pat[pat["EVENT_ID"]=="V04"].shape[0] == 1 # Also only one row
    assert pat[pat["EVENT_ID"]=="SC"]["MCACLCKC"].iloc[0] == 1
    assert pat[pat["EVENT_ID"]=="V04"]["MCACLCKC"].iloc[0] == 0
    assert pat[pat["EVENT_ID"]=="SC"]["NQCOG04"].iloc[0] == 5
    assert pat[pat["EVENT_ID"]=="V04"]["NQCOG04"].iloc[0] == 4
    assert pat[pat["EVENT_ID"]=="SC"]["NQCOG25R"].iloc[0] == 5
    assert pat[pat["EVENT_ID"]=="V04"]["NQCOG25R"].iloc[0] == 4

    # Ensure columns have been deduplicated correctly. NUPSOURC appears in all UPDRS tables
    assert "PAG_NAME" in df.columns.tolist()
    assert "PAG_NAME_x" not in df.columns.tolist()
    assert "PAG_NAME_y" not in df.columns.tolist()
    # Different values get merged into a pipe-separated string
    assert df[(df["PATNO"]=="9999")&(df["EVENT_ID"]=="SC")].iloc[0,:]["PAG_NAME"] == "MOCA|NQCOGNS|NQCOMMS"

def test_empty_dir(caplog, tmp_path):
    df = load_ppmi_non_motor_assessments(tmp_path)

    record = caplog.records[-1] # Last log message
    assert record.levelname == "WARNING"
    assert "No matching non-motor assessment CSV files" in record.message
    assert df.empty

def test_missing_patno(caplog, tmp_path):
    # Set up the empty PATNO file among others
    testfile = "Neuro_QoL__Communication_-_Short_Form"
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    file = tmp_path / SUB_DIR / f"{testfile}_21Test2025.csv"
    tmp = pd.read_csv(file)
    tmp = tmp.drop(columns="PATNO")
    tmp.to_csv(file, index=False)

    df = load_ppmi_non_motor_assessments(tmp_path)

    records = [r for r in caplog.records if testfile in r.message]
    assert len(records) == 2, f"Should be 2 log records for {testfile}: found {len(records)}"

    assert "Loading non-motor assessment file" in records[0].message # normal loading msg
    assert records[1].levelname == "WARNING"
    assert "missing PATNO column, skipping" in records[1].message

    # And the output shouldn't contain NP1SLPD
    assert "NQCOG04" not in df.columns.tolist()
