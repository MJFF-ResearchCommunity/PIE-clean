import sys
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie_clean.utils import *

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"
SUB_DIR = "_Subject_Characteristics"

def test_agg_by_patno_eventid(caplog):
    # Empty df returns empty df
    df = aggregate_by_patno_eventid(pd.DataFrame(), "Subject Characteristics")
    assert df.empty

    # Read some input for testing
    tmp = pd.read_csv(f"{DATA_DIR}/{SUB_DIR}/Age_at_visit_21Test2025.csv")
    tmp["PATNO"] = tmp["PATNO"].astype(str)

    # Missing PATNO or EVENT_ID gives warning and returns original
    tmp2 = tmp.copy().drop(columns="PATNO")
    df = aggregate_by_patno_eventid(tmp2, "Subject Characteristics")
    assert caplog.records[-1].levelname == "WARNING"
    assert "Cannot aggregate" in caplog.records[-1].message
    pd.testing.assert_frame_equal(tmp2, df)

    tmp2 = tmp.copy().drop(columns="EVENT_ID")
    df = aggregate_by_patno_eventid(tmp2, "Subject Characteristics")
    assert caplog.records[-1].levelname == "WARNING"
    assert "Cannot aggregate" in caplog.records[-1].message
    pd.testing.assert_frame_equal(tmp2, df)

    # No duplicates returns a copy of the original
    n_records = len(caplog.records)
    df = aggregate_by_patno_eventid(tmp, "Subject Characteristics")
    assert len(caplog.records) == n_records, f"Call generated more logs than expected"
    pd.testing.assert_frame_equal(tmp, df)

    # Create some duplicate {PATNO, EVENT_ID} rows by copying the first 3
    tmp2 = pd.concat([tmp, tmp.iloc[:3, :]], axis=0).copy()
    tmp2.iloc[-1, 2] = tmp2.iloc[-1, 2]+1.0 # Alter one value
    df = aggregate_by_patno_eventid(tmp2, "Subject Characteristics")
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
    df = aggregate_by_patno_eventid(tmp2, "Subject Characteristics")
    pairs = set()
    for i, row in df.iterrows():
        pairs.add((row["PATNO"], row["EVENT_ID"]))
    assert df.shape[0] == len(pairs), f"Expected {len(pairs)} rows, got {df.shape[0]}"

def test_general_dedup_suffixed_cols(caplog):
    # Empty df returns empty df
    df = general_deduplicate_suffixed_columns(pd.DataFrame())
    assert df.empty

    # No "_x" and "_y" columns returns unaltered df
    tmp = pd.read_csv(f"{DATA_DIR}/{SUB_DIR}/Age_at_visit_21Test2025.csv")
    df = general_deduplicate_suffixed_columns(tmp)
    pd.testing.assert_frame_equal(tmp, df)

    # Now set up the "_x" and "_y" columns
    both_col = "INFODT"
    tmp[f"{both_col}_x"] = ["01/2015"] * tmp.shape[0]
    tmp[f"{both_col}_y"] = ["01/2025"] * tmp.shape[0]
    x_col = "ORIG_ENTRY"
    tmp[f"{x_col}_x"] = ["01/2015"] * tmp.shape[0]
    y_col = "LabX"
    tmp[f"{y_col}_y"] = [1.234] * tmp.shape[0]

    df = general_deduplicate_suffixed_columns(tmp)

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
