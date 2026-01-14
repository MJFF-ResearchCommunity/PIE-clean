import logging
import pytest
import numpy as np
import pandas as pd
from pie_clean import *

logging.getLogger("PIE").setLevel(logging.DEBUG)
DATA_DIR = "tests/test_data"

@pytest.fixture()
def data_dict():
    # Load a dataset with no preprocessing,
    # excluding some of the larger biospecs
    biospec_exclude = ['project_9000', 'project_222', 'project_196']
    logging.getLogger("PIE").setLevel(logging.ERROR)
    data_dict = DataLoader.load(
        data_path=DATA_DIR,
        biospec_exclude=biospec_exclude,
        merge_output=False,
        clean_data=False
    )
    logging.getLogger("PIE").setLevel(logging.DEBUG)
    return data_dict

def test_clean(data_dict):
    clean_dict = DataPreprocessor.clean(data_dict)
    # Test cleaning runs on full dict and returns another dict
    assert isinstance(clean_dict, dict), "Expected a dictionary from cleaning."
    for modality in ALL_MODALITIES:
        assert modality in clean_dict

# Now test the actual cleaning code
def test_clean_features_of_parkinsonism(data_dict):
    clean_df = DataPreprocessor.clean_features_of_parkinsonism(
            data_dict[MEDICAL_HISTORY]["Features_of_Parkinsonism"])

    assert "FEATRIGID" in clean_df
    # There should be no more Uncertain values of 2
    assert (clean_df["FEATBRADY"]!=2).all()
    assert (data_dict[MEDICAL_HISTORY]["Features_of_Parkinsonism"]["FEATRIGID"]==2).any()

    # Try passing NaN as the new value for Uncertain
    clean_df = DataPreprocessor.clean_features_of_parkinsonism(
            data_dict[MEDICAL_HISTORY]["Features_of_Parkinsonism"], uncertain=np.nan)

    # There should be no more Uncertain values of 2
    count = (data_dict[MEDICAL_HISTORY]["Features_of_Parkinsonism"]["FEATRIGID"]==2).sum()
    assert clean_df["FEATRIGID"].isnull().sum() >= count # might be some pre-existing NaNs


def test_clean_gen_physical_exam(data_dict):
    orig_df = data_dict[MEDICAL_HISTORY]["General_Physical_Exam"]
    clean_df = DataPreprocessor.clean_gen_physical_exam(orig_df)

    assert "ABNORM" in clean_df
    # There should be no more Could Not Assess values of 2
    assert (orig_df["ABNORM"]==2).any()
    assert (clean_df["ABNORM"]!=2).all()
    assert (data_dict[MEDICAL_HISTORY]["General_Physical_Exam"]["ABNORM"]==2).any()

    # Try passing NaN as the new Could Not Assess value
    clean_df = DataPreprocessor.clean_gen_physical_exam(
            data_dict[MEDICAL_HISTORY]["General_Physical_Exam"], uncertain=np.nan)

    # There should be no more Uncertain values of 2
    count = (data_dict[MEDICAL_HISTORY]["General_Physical_Exam"]["ABNORM"]==2).sum()
    assert clean_df["ABNORM"].isnull().sum() >= count # might be some pre-existing NaNs


def test_clean_vital_signs(data_dict):
    clean_df = DataPreprocessor.clean_vital_signs(data_dict[MEDICAL_HISTORY]["Vital_Signs"])

    # SYSSUP is in both
    assert "SYSSUP" in clean_df
    assert "SYSSUP" in data_dict[MEDICAL_HISTORY]["Vital_Signs"]
    # The new code and label are only in clean_df
    assert "Sup BP code" in clean_df
    assert "Sup BP code" not in data_dict[MEDICAL_HISTORY]["Vital_Signs"]
    assert "Sup BP label" in clean_df
    assert "Sup BP label" not in data_dict[MEDICAL_HISTORY]["Vital_Signs"]
    assert "Stnd BP code" in clean_df
    assert "Stnd BP label" in clean_df

    # Test specific mappings
    assert clean_df[(clean_df["PATNO"]=="9999")&(clean_df["EVENT_ID"]=="SC")].iloc[0,:]["Sup BP code"] == 0
    assert clean_df[(clean_df["PATNO"]=="9999")&(clean_df["EVENT_ID"]=="BL")].iloc[0,:]["Sup BP code"] == 1
    assert clean_df[(clean_df["PATNO"]=="9999")&(clean_df["EVENT_ID"]=="V01")].iloc[0,:]["Sup BP code"] == 2
    assert clean_df[(clean_df["PATNO"]=="9998")&(clean_df["EVENT_ID"]=="BL")].iloc[0,:]["Stnd BP code"] == 3
    assert clean_df[(clean_df["PATNO"]=="9998")&(clean_df["EVENT_ID"]=="SC")].iloc[0,:]["Sup BP code"] == 4

def test_clean_concomitant_meds(data_dict):
    orig_df = data_dict[MEDICAL_HISTORY]["Concomitant_Medication_Log"]
    clean_df = DataPreprocessor.clean_concomitant_meds(orig_df)
    assert "CMTRT" in clean_df.columns

    assert clean_df["CMTRT"].notnull().all() # All should have names
    assert np.issubdtype(clean_df["STARTDT"], np.datetime64) # Dates should be converted from string
    assert np.issubdtype(clean_df["STOPDT"], np.datetime64) # Dates should be converted from string

    assert clean_df["CMINDC"].notnull().all() # After cleaning, all TEXT is mapped to indication code

    ## Check some specific mappings
    # Aspirin with no indication is for pain
    assert pd.isnull(orig_df[clean_df["CMTRT"]=="ASPIRIN"].iloc[0]["CMINDC_TEXT"])
    assert pd.isnull(orig_df[clean_df["CMTRT"]=="ASPIRIN"].iloc[0]["CMINDC"])
    assert clean_df[clean_df["CMTRT"]=="ASPIRIN"].iloc[0]["CMINDC_TEXT"] == "Pain"
    assert clean_df[clean_df["CMTRT"]=="ASPIRIN"].iloc[0]["CMINDC"] == 17
    # Iron is a supplement
    assert orig_df[clean_df["CMTRT"]=="IRON SUPPLEMENT"].iloc[0]["CMINDC_TEXT"] == "MILD ANEMIA"
    assert pd.isnull(orig_df[clean_df["CMTRT"]=="IRON SUPPLEMENT"].iloc[0]["CMINDC"])
    assert clean_df[clean_df["CMTRT"]=="IRON SUPPLEMENT"].iloc[0]["CMINDC_TEXT"] == "Supplements / Homeopathic Medication"
    assert clean_df[clean_df["CMTRT"]=="IRON SUPPLEMENT"].iloc[0]["CMINDC"] == 22
    # Calcium has a code, so check its text is now set
    assert pd.isnull(orig_df[clean_df["CMTRT"]=="Calcium"].iloc[0]["CMINDC_TEXT"])
    assert orig_df[clean_df["CMTRT"]=="Calcium"].iloc[0]["CMINDC"] == 22
    assert clean_df[clean_df["CMTRT"]=="Calcium"].iloc[0]["CMINDC_TEXT"] == "Supplements / Homeopathic Medication"
    # Amlopidine has text, so check its code is now set
    assert orig_df[clean_df["CMTRT"]=="AMLODIPINE/VALSARTAN 5/320"].iloc[0]["CMINDC_TEXT"] == "HYPERTENSION"
    assert pd.isnull(orig_df[clean_df["CMTRT"]=="AMLODIPINE/VALSARTAN 5/320"].iloc[0]["CMINDC"])
    assert clean_df[clean_df["CMTRT"]=="AMLODIPINE/VALSARTAN 5/320"].iloc[0]["CMINDC_TEXT"] == "Hypertension"
    assert clean_df[clean_df["CMTRT"]=="AMLODIPINE/VALSARTAN 5/320"].iloc[0]["CMINDC"] == 14

def test_clean_ledd_meds(data_dict):
    clean_df = DataPreprocessor.clean_ledd_meds(
            data_dict[MEDICAL_HISTORY]["LEDD_Concomitant_Medication_Log"])
    assert "LEDTRT" in clean_df.columns

    assert clean_df["LEDTRT"].notnull().all() # All should have names
    assert np.issubdtype(clean_df["STARTDT"], np.datetime64) # Dates should be converted from string
    assert np.issubdtype(clean_df["STOPDT"], np.datetime64) # Dates should be converted from string

    # Cleaning should remove some of the nulls (although unfortunately not all)
    assert clean_df["LEDD"].isnull().sum() < \
        data_dict[MEDICAL_HISTORY]["LEDD_Concomitant_Medication_Log"]["LEDD"].isnull().sum()
    # This one can't be converted
    assert pd.isnull(clean_df[clean_df["LEDTRT"]=="LEVODOPA"]["LEDD"].iloc[0])
    # Specific translated values
    assert clean_df[clean_df["LEDTRT"]=="Safinamide"]["LEDD"].iloc[0] == 150
    assert clean_df[clean_df["LEDTRT"]=="Trihexiphenidyl"]["LEDD"].iloc[0] == 100
    assert clean_df[clean_df["LEDTRT"]=="Duopa"]["LEDD"].iloc[0] == 1.1 * 5*3*2
    assert clean_df[clean_df["LEDTRT"]=="Inbrija"]["LEDD"].iloc[0] == 0.69 * 5*3*2
    assert clean_df[clean_df["LEDTRT"]=="Benserazide"]["LEDD"].iloc[0] == 0.85 * 5*3*2
    assert clean_df[clean_df["LEDTRT"]=="Istradefylline"]["LEDD"].iloc[0] == "LD x 0.2"
    assert clean_df[clean_df["LEDTRT"]=="Tolcapone"]["LEDD"].iloc[0] == "LD x 0.5"
    assert clean_df[clean_df["LEDTRT"]=="Entacapone"]["LEDD"].iloc[0] == "LD x 0.33"
    # Complex calculations
    assert clean_df[(clean_df["LEDTRT"]=="Selegiline")&
                    (clean_df["LEDDOSSTR"]=="PO")]["LEDD"].iloc[0] == 10 * 5*3*2
    assert clean_df[(clean_df["LEDTRT"]=="Selegiline")&
                    (clean_df["LEDDOSSTR"]=="Sublingual")]["LEDD"].iloc[0] == 80 * 5*3*2
    assert pd.isnull(clean_df[(clean_df["LEDTRT"]=="Selegiline")&
                              clean_df["LEDDOSSTR"].isnull()]["LEDD"].iloc[0])
    assert clean_df[clean_df["LEDTRT"]=="Apomorphine film"]["LEDD"].iloc[0] == 1.5 * 5*3*2
    assert clean_df[clean_df["LEDTRT"]=="Apomorphine pen"]["LEDD"].iloc[0] == 10 * 5*3*2
    # Various levodopas
    assert clean_df[clean_df["LEDTRT"]=="Levodopa ER"]["LEDD"].iloc[0] == 0.5 * 5*3*2
    assert clean_df[clean_df["LEDTRT"]=="Levodopa CR"]["LEDD"].iloc[0] == 0.75 * 5*3*2

@pytest.mark.skip(reason="Don't recreate every time")
def test_create_concomitant_meds(data_dict):
    DataPreprocessor.create_concomitant_meds(data_dict[MEDICAL_HISTORY]["Concomitant_Medication_Log"])
