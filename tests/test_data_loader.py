import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the DataLoader class and constants
from pie_clean import DataLoader
from pie_clean import (
    SUBJECT_CHARACTERISTICS, MEDICAL_HISTORY, MOTOR_ASSESSMENTS,
    NON_MOTOR_ASSESSMENTS, BIOSPECIMEN
)

# Import the biospecimen loader directly to verify exclusions
from pie_clean.biospecimen_loader import merge_biospecimen_data

def test_data_loader():
    """
    Test basic functionality of the data loading functions.
    """
    # Data location
    data_dir = "./PPMI"
    # Define biospecimen projects to exclude - MUST use exact project names
    biospec_exclude = ['project_9000', 'project_222', 'project_196']

    # Load all data as a dictionary
    all_data_dict = DataLoader.load(
        data_path=data_dir,
        merge_output=False,
        biospec_exclude=biospec_exclude
    )

    assert BIOSPECIMEN in all_data_dict
    # Check if biospecimen data is loaded and verify exclusions
    biospec_data = all_data_dict[BIOSPECIMEN]
    assert isinstance(biospec_data, pd.DataFrame)

if __name__ == "__main__":
    test_data_loader()
