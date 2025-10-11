# IRS 990 Data Pipeline

ETL pipeline for extracting and analyzing IRS Form 990 charitable giving data from XML filings.

## Overview

This pipeline downloads IRS Form 990 tax returns (990, 990-EZ, and 990-PF) and extracts structured data including:
- Grant details and recipient information
- Donor-advised fund statistics
- Independent contractor compensation
- Direct charitable activities
- Program-related investments
- Contributions, grants, and assets

## Prerequisites

- Python 3.12
- 10+ GB free disk space (for raw IRS data)
- Internet connection for downloading IRS files

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/brianallen131/irs-990-data-pipeline.git
cd irs-990-data-pipeline
```

2. **Create a virtual environment**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download IRS Data

Download all available IRS 990 XML files:

```bash
python -m src.02_extractors.download_irs_data
```

Options:
- `--force` - Redownload existing files
- `--sequential` - Use sequential downloads instead of parallel
- `--data-dir PATH` - Custom download directory (default: `data/raw_irs_data`)
- `--inventory` - Show inventory of existing files

Example:
```bash
# Download with custom directory
python -m src.02_extractors.download_irs_data --data-dir ./my_data

# Check what's already downloaded
python -m src.02_extractors.download_irs_data --inventory
```

### 2. Extract Data

Run any of the extraction scripts to process the XML files:

#### Grant Details
```bash
python -m src.02_extractors.grant_details_extractor
```

#### Independent Contractor Compensation
```bash
python -m src.02_extractors.independent_contractor_details_extractor
```

#### Donor-Advised Funds
```bash
python -m src.02_extractors.donor_advised_fund_extractor
```

#### Contributions, Grants & Assets
```bash
python -m src.02_extractors.contributions_grants_assets_extractor
```

#### Direct Charitable Activities
```bash
python -m src.02_extractors.direct_charitable_activity_extractor
```

#### Program-Related Investments
```bash
python -m src.02_extractors.program_related_investments_extractor
```

### Extraction Options

All extractors support these common options:

- `--force-process` - Reprocess files even if output already exists
- `--store-all-filing-data` - Store records even if all fields are null
- `--max-workers N` - Number of parallel workers (default: 1)

Example:
```bash
# Process with 4 parallel workers and force reprocessing
python -m src.02_extractors.grant_details_extractor --max-workers 4 --force-process
```

## Project Structure

```
irs-990-data-pipeline/
├── data/
│   ├── raw_irs_data/              # Downloaded ZIP files
│   └── extracted_irs_data/        # Extracted CSV files by type
├── src/
│   ├── config/
│   │   └── settings.py            # Configuration settings
│   ├── extractors/
│   │   ├── base_extractor.py      # Base extraction class
│   │   ├── download_irs_data.py   # Data downloader
│   │   ├── grant_details_extractor.py
│   │   ├── independent_contractor_details_extractor.py
│   │   ├── donor_advised_fund_extractor.py
│   │   ├── contributions_grants_assets_extractor.py
│   │   ├── direct_charitable_activity_extractor.py
│   │   └── program_related_investments_extractor.py
│   └── utils/
│       ├── logging_config.py      # Logging configuration
│       └── http_utils.py          # HTTP utilities
├── requirements.txt
└── README.md
```

## Output Data

Extracted data is saved as CSV files in `data/extracted_irs_data/` organized by extraction type:

- `grant_details/` - Grant recipient information and amounts
- `independent_contractor_details/` - Contractor compensation data
- `donor_advised_fund_details/` - DAF contributions, grants, and balances
- `contributions_grants_assets/` - High-level financial metrics
- `direct_charitable_activity_details/` - Direct program activities (990-PF only)
- `program_related_investments/` - Program-related investment details (990-PF only)

Each CSV file is named with the pattern: `{type}__{source_zip_file}.csv`

## Data Sources

Data is downloaded from the IRS public dataset:
- **Source**: [IRS Form 990 Data](https://www.irs.gov/charities-non-profits/form-990-series-downloads)
- **Coverage**: E-filed returns from 2010-present
- **Forms**: 990, 990-EZ, 990-PF
- **Format**: XML files in ZIP archives

## Common Fields

All extracted datasets include these metadata fields:
- `filing_number` - Unique IRS filing identifier
- `zip_name` - Source ZIP file name
- `ein` - Employer Identification Number
- `tax_yr` - Tax year of filing
- `form` - Form type (990, 990EZ, or 990PF)

## Performance Tips

1. **Parallel Processing**: Use `--max-workers` to speed up extraction
   ```bash
   python -m src.02_extractors.grant_details_extractor --max-workers 8
   ```

2. **Skip Existing Files**: By default, extractors skip already-processed files. Use `--force-process` only when needed.

3. **Disk Space**: Each ZIP file is 50-200 MB. Monitor disk usage when downloading all files.

4. **Memory**: Processing large ZIP files may require 2-4 GB RAM per worker.

## Troubleshooting

### Download Issues
- **Slow downloads**: Try `--sequential` flag or reduce `--max-workers`
- **Connection timeouts**: Run downloader again; it will resume where it left off

### Extraction Issues
- **Missing data**: Not all forms contain all fields. Empty fields are expected.
- **XML parsing errors**: Some files may have malformed XML. These are logged and skipped.
- **Memory errors**: Reduce `--max-workers` or process fewer files at once

### Check Logs
All operations are logged. Check console output for detailed progress and error messages.

## Development

### Running Tests
```bash
pytest
```

### Code Style
Follow PEP 8 guidelines. The codebase uses:
- Type hints where applicable
- Descriptive variable names
- Docstrings for all public methods

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- IRS for providing public access to 990 data
- Open Data for Nonprofit Research community for documentation and tools

## Contact

Brian Allen - [GitHub](https://github.com/brianallen131)

Project Link: [https://github.com/brianallen131/irs-990-data-pipeline](https://github.com/brianallen131/irs-990-data-pipeline)

## Additional Resources

- [IRS Form 990 Documentation](https://www.irs.gov/forms-pubs/about-form-990)
- [Form 990 XML Schema](https://www.irs.gov/e-file-providers/current-valid-xml-schemas-and-business-rules-for-exempt-organizations-modernized-e-file)
- [Nonprofit Open Data Collective](https://nonprofit-open-data-collective.github.io/)
