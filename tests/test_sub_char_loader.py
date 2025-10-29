import sys
import shutil
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie_clean.sub_char_loader import (
    load_ppmi_subject_characteristics, _general_deduplicate_suffixed_columns,
    _aggregate_by_patno_eventid, FILE_PREFIXES
)

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"
SUB_DIR = "_Subject_Characteristics" 

load_msg = f"Loading subject characteristics file: {DATA_DIR}/{SUB_DIR}"

def test_load_ppmi_subject_characteristics(caplog):
    df = load_ppmi_subject_characteristics(DATA_DIR)

    # We expect to see certain files, which are handled in specific ways
    # The logger output clarifies how they were handled
    # First, just check that we're looking for everything
    for filename in FILE_PREFIXES:
        assert filename in caplog.text

    # Now pick out some specifics
    for record in caplog.records:
        if "Age_at_visit" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/Age_at_visit" in record.message
        elif "Family_History" in record.message:
            # One and only one message, for loading
            assert f"{load_msg}/Family_History" in record.message
        elif "Participant_Status" in record.message:
            # Two logs: one for loading, and one for merging
            assert f"{load_msg}/Participant_Status" in record.message or \
                    "on PATNO only (it lacks EVENT_ID)" in record.message
            # And merging must be in there somewhere
            assert "on PATNO only (it lacks EVENT_ID)" in caplog.text

    # We expect to see something from every table merged together in the output
    assert "PATNO" in df.columns
    assert "EVENT_ID" in df.columns
    assert "AGE_AT_VISIT" in df.columns
    assert "ANYFAMPD" in df.columns
    assert "ENRLGBA" in df.columns

    # Ensure that EVENT_IDs across tables have been properly merged
    pat = df[df["PATNO"]=="9999"]
    assert "V01" in pat["EVENT_ID"].tolist() # But no data other than AGE_AT_VISIT
    assert "V04" in pat["EVENT_ID"].tolist() # AGE_AT_VISIT and Fam History
    assert pat[pat["EVENT_ID"]=="V01"].shape[0] == 1 # Only one row
    assert pat[pat["EVENT_ID"]=="V04"].shape[0] == 1 # Also only one row
    assert not pd.isnull(pat[pat["EVENT_ID"]=="V01"]["AGE_AT_VISIT"].iloc[0])
    assert pd.isnull(pat[pat["EVENT_ID"]=="V01"]["ANYFAMPD"].iloc[0])
    assert not pd.isnull(pat[pat["EVENT_ID"]=="V04"]["AGE_AT_VISIT"].iloc[0])
    assert not pd.isnull(pat[pat["EVENT_ID"]=="V04"]["ANYFAMPD"].iloc[0])

def test_empty_dir(caplog, tmp_path):
    df = load_ppmi_subject_characteristics(tmp_path)

    record = caplog.records[-1] # Last log message
    assert record.levelname == "WARNING"
    assert "No matching subject characteristics" in record.message
    assert df.empty

def test_missing_patno(caplog, tmp_path):
    # Set up the empty PATNO file among others
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    file = tmp_path / SUB_DIR / "Age_at_visit_21Test2025.csv"
    tmp = pd.read_csv(file)
    tmp = tmp.drop(columns="PATNO")
    tmp.to_csv(file, index=False)

    df = load_ppmi_subject_characteristics(tmp_path)

    records = [r for r in caplog.records if "Age_at_visit" in r.message]
    assert len(records) == 2, f"Should be 2 log records for Age_at_visit: found {len(records)}"

    assert "Loading subject characteristics file" in records[0].message # normal loading msg
    assert records[1].levelname == "WARNING"
    assert "missing PATNO column, skipping" in records[1].message

    # And the output shouldn't contain AGE_AT_VISIT
    assert "AGE_AT_VISIT" not in df.columns.tolist()

def test_general_dedup_suffixed_cols(caplog):
    # Empty df returns empty df
    df = _general_deduplicate_suffixed_columns(pd.DataFrame())
    assert df.empty

    # No "_x" and "_y" columns returns unaltered df
    tmp = pd.read_csv(f"{DATA_DIR}/{SUB_DIR}/Age_at_visit_21Test2025.csv")
    df = _general_deduplicate_suffixed_columns(tmp)
    pd.testing.assert_frame_equal(tmp, df)

    # Now set up the "_x" and "_y" columns
    both_col = "INFODT"
    tmp[f"{both_col}_x"] = ["01/2015"] * tmp.shape[0]
    tmp[f"{both_col}_y"] = ["01/2025"] * tmp.shape[0]
    x_col = "ORIG_ENTRY"
    tmp[f"{x_col}_x"] = ["01/2015"] * tmp.shape[0]
    y_col = "LabX"
    tmp[f"{y_col}_y"] = [1.234] * tmp.shape[0]

    df = _general_deduplicate_suffixed_columns(tmp)

    for record in caplog.records:
        if "Deduplicating suffixed columns for bases" in record.message:
            assert both_col in record.message
            assert x_col in record.message
            assert y_col in record.message
        elif "Combining " in record.message:
            assert f"{both_col}_x and {both_col}_y" in record.message
        elif "Renaming " in record.message:
            assert f"{x_col}_x to {x_col}" in record.message or \
                   f"{y_col}_y to {y_col}" in record.message

    assert both_col in df.columns.tolist()
    assert x_col in df.columns.tolist()
    assert y_col in df.columns.tolist()
    assert (df[both_col] == "01/2015|01/2025").all()
    assert (df[x_col] == "01/2015").all()
    assert (df[y_col] == 1.234).all()

def test_agg_by_patno_eventid(caplog):
    # Empty df returns empty df
    df = _aggregate_by_patno_eventid(pd.DataFrame())
    assert df.empty

    # Read some input for testing
    tmp = pd.read_csv(f"{DATA_DIR}/{SUB_DIR}/Age_at_visit_21Test2025.csv")
    tmp["PATNO"] = tmp["PATNO"].astype(str)

    # Missing PATNO or EVENT_ID gives warning and returns original
    tmp2 = tmp.copy().drop(columns="PATNO")
    df = _aggregate_by_patno_eventid(tmp2)
    assert caplog.records[-1].levelname == "WARNING"
    assert "Cannot aggregate" in caplog.records[-1].message
    pd.testing.assert_frame_equal(tmp2, df)

    tmp2 = tmp.copy().drop(columns="EVENT_ID")
    df = _aggregate_by_patno_eventid(tmp2)
    assert caplog.records[-1].levelname == "WARNING"
    assert "Cannot aggregate" in caplog.records[-1].message
    pd.testing.assert_frame_equal(tmp2, df)

    # No duplicates returns a copy of the original
    n_records = len(caplog.records)
    df = _aggregate_by_patno_eventid(tmp)
    assert len(caplog.records) == n_records, f"Call generated more logs than expected"
    pd.testing.assert_frame_equal(tmp, df)

    # Create some duplicate {PATNO, EVENT_ID} rows by copying the first 3
    tmp2 = pd.concat([tmp, tmp.iloc[:3, :]], axis=0).copy()
    tmp2.iloc[-1, 2] = tmp2.iloc[-1, 2]+1.0 # Alter one value
    df = _aggregate_by_patno_eventid(tmp2)
    assert "Consolidating rows with duplicate" in caplog.text
    # The first two duplicates should be as-is, although re-indexed
    row = tmp2.iloc[0, :]
    test = df[(df["PATNO"]==row["PATNO"])&(df["EVENT_ID"]==row["EVENT_ID"])]
    assert test.shape[0] == 1 # Duplicates merged into one
    pd.testing.assert_series_equal(row, test.iloc[0, :], check_names=False)
    row = tmp2.iloc[1, :]
    test = df[(df["PATNO"]==row["PATNO"])&(df["EVENT_ID"]==row["EVENT_ID"])]
    assert test.shape[0] == 1 # Duplicates merged into one
    pd.testing.assert_series_equal(row, test.iloc[0, :], check_names=False)
    # The 3rd row was updated though
    row = tmp2.iloc[2, :]
    test = df[(df["PATNO"]==row["PATNO"])&(df["EVENT_ID"]==row["EVENT_ID"])]
    assert test.shape[0] == 1 # Duplicates merged into one, still
    assert row["PATNO"] == test["PATNO"].iloc[0]
    assert row["EVENT_ID"] == test["EVENT_ID"].iloc[0]
    assert test["AGE_AT_VISIT"].iloc[0] == f"{row['AGE_AT_VISIT']}|{row['AGE_AT_VISIT']+1.0}"
    # And log output contains more info
    assert "Summary of pipe-separated columns" in caplog.records[-3].message
    assert "Column 'AGE_AT_VISIT'" in caplog.records[-2].message
    assert "values were ['47.4', '48.4']" in caplog.records[-1].message

    # Nothing but PATNO and EVENT_ID: Remove the dups
    tmp2 = tmp2.drop(columns="AGE_AT_VISIT")
    assert len(tmp2.columns) == 2
    df = _aggregate_by_patno_eventid(tmp2)
    pairs = set()
    for i, row in df.iterrows():
        pairs.add((row["PATNO"], row["EVENT_ID"]))
    assert df.shape[0] == len(pairs), f"Expected {len(pairs)} rows, got {df.shape[0]}"
