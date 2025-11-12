import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the DataLoader class and constants
from pie_clean import DataLoader
from pie_clean import ALL_MODALITIES, BIOSPECIMEN

DATA_DIR = "tests/test_data"

# Import the biospecimen loader directly to verify exclusions
from pie_clean.biospecimen_loader import merge_biospecimen_data

def test_data_loader(caplog):
    """
    Test basic functionality of the data loading functions.
    """
    # Define biospecimen projects to exclude - MUST use exact project names
    biospec_exclude = ['project_9000', 'project_222', 'project_196']

    # Load all data as a dictionary, with clean_data=False to avoid preprocessing
    all_data_dict = DataLoader.load(
        data_path=DATA_DIR,
        merge_output=False,
        clean_data=False,
        biospec_exclude=biospec_exclude
    )
    assert isinstance(all_data_dict, dict)
    for modality in ALL_MODALITIES:
        assert modality in all_data_dict
    # Check if biospecimen data is loaded and verify exclusions
    biospec_data = all_data_dict[BIOSPECIMEN]
    assert isinstance(biospec_data, pd.DataFrame)
    for col in biospec_data.columns:
        # The excluded ones should not exist
        for excl in biospec_exclude:
            assert not col.startswith(excl)
    # And the ones which are included by default should exist
    assert any([col.startswith("project_177") for col in biospec_data.columns.tolist()])
    # Loading generates a lot of logging. Instead of strict limits, check for reasonable bounds.
    # Should be around 77
    infos = [r for r in caplog.records if r.levelname=="INFO"]
    n = len(infos)
    assert n > 60, f"Suspiciously little logging generated: {n} records"
    assert n < 100, f"Suspiciously lots of logging generated: {n} records"
    caplog.clear() # Reset for future calls

    # Now load the same data, but merge
    all_data = DataLoader.load(
        data_path=DATA_DIR,
        merge_output=True,
        clean_data=False,
        biospec_exclude=biospec_exclude
    )
    assert isinstance(all_data, pd.DataFrame)
    # Check for columns from individual modalities
    assert "AGE_AT_VISIT" in all_data.columns.tolist() # From Age at Visit in Subject Chars
    assert "TEMPC" in all_data.columns.tolist() # From Vital Signs in Medical History
    assert "NP2WALK" in all_data.columns.tolist() # From MCD-UPDRS pt 2 in Motor Exams
    assert "NQCOG25R" in all_data.columns.tolist() # From Neuro QoL in Non-motor Exams
    assert "project_151_pQTL_CSF_151_10000-28_3" in all_data.columns.tolist() # From Biospecimens
    # Verify exclusions in biospecimens
    for col in all_data.columns:
        for excl in biospec_exclude:
            assert not col.startswith(excl)
    # Loading generates a lot of logging. Instead of strict limits, check for reasonable bounds
    # Should be around 90
    infos = [r for r in caplog.records if r.levelname=="INFO"]
    n = len(infos)
    assert n > 60, f"Suspiciously little logging generated: {n} records"
    assert n < 100, f"Suspiciously lots of logging generated: {n} records"


if __name__ == "__main__":
    test_data_loader()
