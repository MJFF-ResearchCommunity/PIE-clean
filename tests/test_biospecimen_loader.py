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

    # Test when the directory is empty
    df = load_project_151_pQTL_CSF(tmp_path)
    assert caplog.records[-1].levelname == "WARNING"
    assert "No non-batch-corrected Project_151_pQTL_in_CSF files" in caplog.records[-1].message
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

def test_load_metabolomic_lrrk2(caplog, tmp_path):
    # First, skip the CSF files
    df = load_metabolomic_lrrk2(f"{DATA_DIR}/{SUB_DIR}", include_csf=False)
    # Logging shows that the files were found correctly
    assert "Successfully processed Metabolomic" in caplog.records[-1].message
    # Logging shows the data was pivoted into wide format
    assert "2 rows, 8 columns" in caplog.records[-1].message
    assert df.shape[0] == 2 # ...and the output matches the logging
    # PATNOs from each file have been merged together (ints here)
    assert 9999 in df["PATNO"].tolist()
    assert 9998 in df["PATNO"].tolist()
    # Test IDs have become column names, with "LRRK2_" prepended
    assert "LRRK2_MZ100.08_RT586.24_pos" in df.columns.tolist()
    # But the CSF tests are not included
    assert "LRRK2_(3-O-sulfo)GalCer(d18:1/16:0)" not in df.columns.tolist()
    # Others have been dropped
    assert "UNITS" not in df.columns.tolist()
    # But SEX was kept
    assert "SEX" in df.columns.tolist()
    # CLINICAL_EVENT has been renamed to EVENT_ID
    assert "CLINICAL_EVENT" not in df.columns.tolist()
    assert "EVENT_ID" in df.columns.tolist()
    assert (df["EVENT_ID"] == "V04").all(), f"Non-V04 EVENT_IDs included: {df['EVENT_ID'].unique()}"
    # And data is complete
    assert not df.isnull().any().any()

    # Next, test the CSF switch defaults to True
    df = load_metabolomic_lrrk2(f"{DATA_DIR}/{SUB_DIR}")
    # Logging shows that the files were found correctly
    assert "Successfully processed Metabolomic" in caplog.records[-1].message
    # Logging shows more rows and more columns
    assert "4 rows, 9 columns" in caplog.records[-1].message
    assert df.shape[0] == 4 # ...and the output matches the logging
    # Now the CSF tests are included
    assert "LRRK2_(3-O-sulfo)GalCer(d18:1/16:0)" in df.columns.tolist()
    # And we have V06 events as well as the original V04s
    assert "V04" in df["EVENT_ID"].tolist()
    assert "V06" in df["EVENT_ID"].tolist()
    # V06 non-CSF tests are null
    assert df[df["EVENT_ID"]=="V04"]["LRRK2_MZ100.08_RT586.24_pos"].notnull().all()
    assert df[df["EVENT_ID"]=="V06"]["LRRK2_MZ100.08_RT586.24_pos"].isnull().all()

    # Test when the directory is empty
    df = load_metabolomic_lrrk2(tmp_path)
    assert caplog.records[-1].levelname == "WARNING"
    assert "No Metabolomic_Analysis_of_LRRK2 files" in caplog.records[-1].message
    assert df.empty

    # Test when required column is missing: set up missing PATNO files
    testfile = "Metabolomic_Analysis_of_LRRK2_PD"
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    for n in [1, 2]:
        file = tmp_path / SUB_DIR / f"{testfile}_{n}_of_5_21Test2025.csv"
        tmp = pd.read_csv(file)
        tmp = tmp.drop(columns="PATNO")
        tmp.to_csv(file, index=False)

    df = load_metabolomic_lrrk2(tmp_path, include_csf=False)
    assert caplog.records[-1].levelname == "ERROR"
    assert "Required column PATNO not found" in caplog.records[-1].message
    assert df.empty

def test_load_project_9000(caplog, tmp_path):
    # Normal loading
    df = load_project_9000(f"{DATA_DIR}/{SUB_DIR}")
    # Logging shows that the files were found correctly
    assert "Successfully processed Project 9000" in caplog.records[-1].message
    # Logging shows the data was pivoted into wide format
    assert "4 rows, 8 columns" in caplog.records[-1].message
    assert df.shape[0] == 4 # ...and the output matches the logging
    # PATNOs from each file have been merged together (strings here)
    assert "9999" in df["PATNO"].tolist()
    assert "9998" in df["PATNO"].tolist()
    # Test IDs have become column names, with "9000_" and the tissue prepended
    assert "9000_CSF_A1L4H1_SSC5D_LOD" in df.columns.tolist()
    assert "9000_Plasma_A1L4H1_SSC5D_LOD" in df.columns.tolist()
    # Others have been dropped
    assert "OLINKID" not in df.columns.tolist()
    # Project 9000 has EVENT_ID from the start
    assert "BL" in df["EVENT_ID"].tolist()
    assert "V04" in df["EVENT_ID"].tolist()
    # And data is complete
    assert not df.isnull().any().any()

    # Test when the directory is empty
    df = load_project_9000(tmp_path)
    assert caplog.records[-1].levelname == "WARNING"
    assert "No PPMI_Project_9000 files" in caplog.records[-1].message
    assert df.empty

    # Test when required column is missing: set up missing PATNO files
    testfile = "PPMI_Project_9000"
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    for n in ["CSF", "Plasma"]:
        file = tmp_path / SUB_DIR / f"{testfile}_{n}_Cardio_NPX_21Test2025.csv"
        tmp = pd.read_csv(file)
        tmp = tmp.drop(columns="PATNO")
        tmp.to_csv(file, index=False)

    df = load_project_9000(tmp_path)
    # Not the last records, but repeated for each failure to find a PATNO
    assert caplog.records[-3].levelname == "ERROR"
    assert "Required columns ['PATNO'] not found" in caplog.records[-3].message
    assert caplog.records[-5].levelname == "ERROR"
    assert "Required columns ['PATNO'] not found" in caplog.records[-5].message
    # And the last record should note that no PATNOs were found at all
    assert caplog.records[-1].levelname == "ERROR"
    assert "Required column PATNO not found" in caplog.records[-1].message
    assert df.empty

def test_load_project_222(caplog, tmp_path):
    # Normal loading
    df = load_project_222(f"{DATA_DIR}/{SUB_DIR}")
    # Logging shows that the files were found correctly
    assert "Successfully processed Project 222" in caplog.records[-1].message
    # Logging shows the data was pivoted into wide format
    assert "4 rows, 14 columns" in caplog.records[-1].message
    assert df.shape[0] == 4 # ...and the output matches the logging
    # PATNOs from each file have been merged together (strings here)
    assert "9999" in df["PATNO"].tolist()
    assert "9998" in df["PATNO"].tolist()
    # Test IDs have become column names, with "222_" and the tissue prepended
    assert "222_CSF_P16860_NPPB_LOD" in df.columns.tolist()
    assert "222_Plasma_O43854_EDIL3_LOD" in df.columns.tolist()
    # Others have been dropped
    assert "OLINKID" not in df.columns.tolist()
    # Project 222 has EVENT_ID from the start
    assert "BL" in df["EVENT_ID"].tolist()
    assert "V04" in df["EVENT_ID"].tolist()
    # And data is complete
    assert not df.isnull().any().any()

    # Test when the directory is empty
    df = load_project_222(tmp_path)
    assert caplog.records[-1].levelname == "WARNING"
    assert "No PPMI_Project_222 files" in caplog.records[-1].message
    assert df.empty

    # Test when required column is missing: set up missing PATNO files
    testfile = "PPMI_Project_222"
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    for n in ["CSF", "Plasma"]:
        file = tmp_path / SUB_DIR / f"{testfile}_{n}_Cardio_NPX_21Test2025.csv"
        tmp = pd.read_csv(file)
        tmp = tmp.drop(columns="PATNO")
        tmp.to_csv(file, index=False)

    df = load_project_222(tmp_path)
    # Not the last records, but repeated for each failure to find a PATNO
    assert caplog.records[-3].levelname == "ERROR"
    assert "Required columns ['PATNO'] not found" in caplog.records[-3].message
    assert caplog.records[-5].levelname == "ERROR"
    assert "Required columns ['PATNO'] not found" in caplog.records[-5].message
    # And the last record should note that no PATNOs were found at all
    assert caplog.records[-1].levelname == "ERROR"
    assert "Required column PATNO not found" in caplog.records[-1].message
    assert df.empty

def test_load_project_196(caplog, tmp_path):
    # Normal loading
    df = load_project_196(f"{DATA_DIR}/{SUB_DIR}")
    # Logging shows that the files were found correctly
    assert "Successfully processed Project 196" in caplog.records[-1].message
    # Logging shows the data was pivoted into wide format
    assert "4 rows, 24 columns" in caplog.records[-1].message
    assert df.shape[0] == 4 # ...and the output matches the logging
    # PATNOs from each file have been merged together (strings here)
    assert "9999" in df["PATNO"].tolist()
    assert "9998" in df["PATNO"].tolist()
    # Test IDs have become column names, with "196_" and the tissue prepended
    assert "196_Plasma_O00584_RNASET2_COUNT" in df.columns.tolist()
    assert "196_CSF_O00584_RNASET2_COUNT" in df.columns.tolist()
    # These were both COUNTs files: we should have NPX columns too
    assert "196_Plasma_O43854_EDIL3_NPX" in df.columns.tolist()
    # Others have been dropped
    assert "OLINKID" not in df.columns.tolist()
    # Project 196 has EVENT_ID from the start
    assert "BL" in df["EVENT_ID"].tolist()
    assert "V04" in df["EVENT_ID"].tolist()
    # And data is complete
    assert not df.isnull().any().any()

    # Test when the directory is empty
    df = load_project_196(tmp_path)
    assert caplog.records[-1].levelname == "WARNING"
    assert "No PPMI_Project_196 files" in caplog.records[-1].message
    assert df.empty

    # Test when required column is missing: set up missing PATNO files
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    for testfile in [
            "PPMI_Project_196_CSF_Cardio_Counts_21Test2025.csv",
            "PPMI_Project_196_Plasma_CARDIO_Counts_21Test2025.csv",
            "PPMI_Project_196_Plasma_Cardio_NPX_21Test2025.csv"]:
        file = tmp_path / SUB_DIR / testfile
        tmp = pd.read_csv(file)
        tmp = tmp.drop(columns="PATNO")
        tmp.to_csv(file, index=False)

    # Clear the caplog so we can be sure of what was generated by this call
    caplog.clear()
    df = load_project_196(tmp_path)
    # Not the last records, but repeated for each failure to find a PATNO
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 8
    err_type = [r for r in errors if "Error reading PATNO/EVENT_ID" in r.message]
    assert len(err_type) == 3
    err_type = [r for r in errors if "Required columns ['PATNO'] not found" in r.message]
    assert len(err_type) == 3
    # And the last record should note that no PATNOs were found at all
    assert caplog.records[-1].levelname == "ERROR"
    assert "Error merging NPX and Counts data" in caplog.records[-1].message
    assert df.empty

def test_load_project_177(caplog, tmp_path):
    # Defaults to batch_corrected=False
    df = load_project_177_untargeted_proteomics(f"{DATA_DIR}/{SUB_DIR}")
    # Logging shows that the files were found correctly
    assert "Successfully processed Project 177" in caplog.records[-1].message
    # Logging shows the data was pivoted into wide format
    assert "2 rows, 8 columns" in caplog.records[-1].message
    assert df.shape[0] == 2 # ...and the output matches the logging
    # PATNOs from each file have been merged together (ints here)
    assert 9999 in df["PATNO"].tolist()
    assert 9998 in df["PATNO"].tolist()
    # Test IDs have become column names, with "177_" prepended
    assert "177_P55058" in df.columns.tolist()
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

    # Test when the directory is empty
    df = load_project_177_untargeted_proteomics(tmp_path)
    assert caplog.records[-1].levelname == "WARNING"
    assert "No PPMI_Project_177 files" in caplog.records[-1].message
    assert df.empty

    # Test when required column is missing: set up missing PATNO files
    testfile = "PPMI_Project_1777_Untargeted_Proteomics_21TestOct.csv"
    shutil.copytree(f"{DATA_DIR}/{SUB_DIR}", tmp_path / SUB_DIR, dirs_exist_ok=True)
    file = tmp_path / SUB_DIR / testfile
    tmp = pd.read_csv(file)
    tmp = tmp.drop(columns="PATNO")
    tmp.to_csv(file, index=False)

    df = load_project_177_untargeted_proteomics(tmp_path)
    assert caplog.records[-1].levelname == "ERROR"
    assert "Required column PATNO not found" in caplog.records[-1].message
    assert df.empty


