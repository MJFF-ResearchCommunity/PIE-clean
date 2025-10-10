# Parkinson's Insight Engine Data Cleaner (PIE-clean)

## Overview
Parkinson's Insight Engine Data Cleaner (PIE-clean) is a Python-based data cleaning pipeline designed for researchers working with the Michael J. Fox Foundation's Parkinson's Progression Markers Initiative (PPMI) dataset and other MJFF data. PIE-clean pairs with [PIE for end-to-end machine learning data analysis](https://github.com/MJFF-ResearchCommunity/PIE).

If you want to perform machine learning on the PPMI dataset, it is recommended that you work with [the PIE package](https://github.com/MJFF-ResearchCommunity/PIE) directly. PIE integrates PIE-clean seamlessly, and will enable data loading and cleaning as part of the pipeline.

If you want to explore and clean the PPMI data, it is recommended that you work with this package, PIE-clean. PIE-clean focuses on loading, merging, and preprocessing the data so that you can explore relationships between parameters across modalities.

## Getting Started

### Prerequisites
- Python 3.8 or later.
- Required dependencies can be installed from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Installation
Clone the repository and install the PIE package. For development, use the editable "`-e`" flag.
```bash
git clone https://github.com/MJFF-ResearchCommunity/PIE-clean.git
cd PIE-clean
pip install -e .
```

### Data Setup
1.  **Download PPMI Data**: You must [apply for access to the PPMI data](https://www.ppmi-info.org/access-data-specimens/download-data).
2.  **Organize Data**: Create a directory named `PPMI` at the root of the cloned PIE-clean repository. Download the individual study data folders from LONI and place them inside the `PPMI` directory. The structure should look like this:
    ```plaintext
    PIE-clean/
    ├── PPMI/
    │   ├── _Subject_Characteristics/
    │   ├── Biospecimen/
    │   ├── Motor___MDS-UPDRS/
    │   ├── Non-motor_Assessments/
    │   ├── Medical_History/
    │   └── ... (other data folders)
    ├── pie_clean/
    └── ... (other project files)
    ```

## How to Use PIE-clean

There are two sources of documentation to help you understand the PIE-clean functionality:
- Please read through the `documentation` directory for API details and code snippets for loading specific data types, expecially the more complex modalities such as biospecimens.
- Please look in the `notebooks` directory for examples of how to use PIE-clean in your workflows.

## Running Tests
To verify your setup and ensure all components are working correctly, you can run the test suite.

```bash
pytest tests
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Make your changes.
4. Add or update tests for your changes.
5. Ensure the full test suite passes: `pytest tests/`.
6. Commit your changes and create a pull request.

## Contributors
- Cameron Hamilton (originator)
- Victoria Catterson

## License
This project is licensed under the MIT License.

## Contact
If you have any questions or suggestions, please don't hesitate to contact vic [at] cowlet [dot] org.
