import sys
import shutil
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie_clean.biospecimen_loader import *

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"
SUB_DIR = "Biospecimen"

def test_load_project_151(caplog, tmp_path):
    # Defaults to batch_corrected=False
    df = load_project_151_pQTL_CSF(f"{DATA_DIR}/{SUB_DIR}")

    # Logging shows that the files were found correctly
    assert "Successfully processed Project_151" in caplog.records[-1].message
    # Logging shows the data was pivoted into wide format
    assert "2 rows, 8 columns" in caplog.records[-1].message
    assert df.shape[0] == 2 # ...and the output matches the logging
    # PATNOs from each file have been merged together (ints here)
    assert 9999 in df["PATNO"].tolist()
    assert 9998 in df["PATNO"].tolist()
    # Test IDs have become column names, with "151_" prepended
    assert "151_10001-7_3" in df.columns.tolist()
    # Others have been dropped
    assert "PLATEID" not in df.columns.tolist()
    # But SEX was kept
    assert "SEX" in df.columns.tolist()
    # CLINICAL_EVENT has been renamed to EVENT_ID
    assert "CLINICAL_EVENT" not in df.columns.tolist()
    assert "EVENT_ID" in df.columns.tolist()
    assert df["EVENT_ID"].iloc[0] == "BL"
    # And data is complete
    assert not df.isnull().any().any()

    # Next, we haven't mocked the files for batch_corrected=True
    df = load_project_151_pQTL_CSF(f"{DATA_DIR}/{SUB_DIR}", batch_corrected=True)
    assert caplog.records[-1].levelname == "WARNING"
    assert "No batch-corrected Project_151" in caplog.records[-1].message
    assert df.empty

    # Test when required column is missing: set up missing PATNO files
    testfile = "Project_151_pQTL_in_CSF"
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    for n in [1, 2]:
        file = tmp_path / SUB_DIR / f"{testfile}_{n}_of_6_21Test2025.csv"
        tmp = pd.read_csv(file)
        tmp = tmp.drop(columns="PATNO")
        tmp.to_csv(file, index=False)

    df = load_project_151_pQTL_CSF(tmp_path)
    assert caplog.records[-1].levelname == "ERROR"
    assert "Required column PATNO not found" in caplog.records[-1].message
    assert df.empty


