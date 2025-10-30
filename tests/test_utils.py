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
