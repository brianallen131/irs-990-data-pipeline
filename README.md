# IRS 990 Data Pipeline

A comprehensive ETL pipeline for extracting, processing, and analyzing IRS Form 990 charitable giving data. This project enables researchers and analysts to work with nonprofit tax return data, including grant details, organizational information, and donor-advised funds.

## 🎯 Features

- **Automated Data Download**: Scrapes and downloads IRS Form 990 data files from official sources
- **XML Processing**: Extracts structured data from IRS XML files with support for Forms 990, 990-EZ, and 990-PF
- **Grant Matching**: Advanced fuzzy matching to link grant recipients with their EINs using the IRS Business Master File
- **Machine Learning**: Two-tower neural network for donor-recipient recommendations
- **Parallel Processing**: Multi-threaded extraction and processing for improved performance
- **Comprehensive Data**: Extracts grants, contributions, assets, DAF details, contractor info, and more

## 📊 Data Pipeline

```
1. Download → 2. Extract → 3. Compile → 4. Clean → 5. Analyze/ML
```

### Pipeline Stages

1. **Download** (`01_downloaders/`)
   - Downloads IRS Form 990 ZIP files from official sources
   - Downloads IRS Business Master File (BMF) data for EIN matching

2. **Extract** (`02_extractors/`)
   - Parses XML files from ZIP archives
   - Extracts structured data into CSV files per ZIP archive
   - Supports multiple data types: filing details, grants, contributions, DAF info, etc.

3. **Compile** (`03_compilers/`)
   - Aggregates CSV files into unified datasets
   - Deduplicates records
   - Creates master organization file

4. **Clean** (`04_cleaners/`)
   - Fuzzy matching for grant recipient EINs
   - Data standardization and quality improvements

5. **Machine Learning** (`05_ml/`)
   - Two-tower recommendation system
   - Donor-to-recipient matching (D2R)
   - Recipient-to-donor matching (R2D)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 20+ GB free disk space (for data storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/brianallen131/irs-990-data-pipeline.git
cd irs-990-data-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Download IRS Data

```bash
# Download Form 990 files
python download_irs_data.py

# Download Business Master File
python download_bmf_data.py
```

#### 2. Extract Data

```bash
# Extract grant details
python -m src.extractors.grant_details_extractor

# Extract organization details
python -m src.extractors.filing_details_extractor

# Extract with parallel processing (4 workers)
python -m src.extractors.grant_details_extractor --max-workers 4
```

#### 3. Compile Data

```bash
# Compile organization details (run this first)
python filing_details_compiler.py

# Compile grant details
python grant_details_compiler.py

# Compile other datasets
python independent_contractor_details_compiler.py
python program_related_investments_compiler.py
python direct_charitable_activity_details_compiler.py
```

#### 4. Clean Data

```bash
# Match grant recipients to EINs using fuzzy matching
python grant_details_recipients_inferred.py
```

#### 5. Train ML Model

```bash
# Train two-tower recommendation model
python grant_ttsn_recommender.py

# Get recommendations for a donor
python grant_ttsn_inference_D2R.py --donor_ein 123456789 --top_k 10

# Find potential donors for a nonprofit
python grant_ttsn_inference_R2D.py --receiver_ein 987654321 --top_k 20
```

## 📁 Project Structure

```
irs-990-data-pipeline/
├── 01_downloaders/
│   ├── download_irs_data.py          # Download Form 990 files
│   └── download_bmf_data.py          # Download BMF data
├── 02_extractors/
│   ├── base_extractor.py             # Base extraction class
│   ├── filing_details_extractor.py   # Organization info
│   ├── grant_details_extractor.py    # Grant data
│   ├── contributions_grants_assets_extractor.py
│   └── ...                           # Other extractors
├── 03_compilers/
│   ├── filing_details_compiler.py    # Main organization file
│   ├── grant_details_compiler.py     # Compiled grants
│   └── ...                           # Other compilers
├── 04_cleaners/
│   └── grant_details_recipients_inferred.py  # EIN matching
├── 05_ml/
│   ├── grant_ttsn_recommender.py     # Train model
│   ├── grant_ttsn_inference_D2R.py   # Donor recommendations
│   └── grant_ttsn_inference_R2D.py   # Find donors
├── src/
│   ├── config/
│   │   └── settings.py               # Configuration
│   └── utils/
│       ├── logging_config.py         # Logging setup
│       └── http_utils.py             # HTTP utilities
├── data/                             # Data directory (created automatically)
│   ├── raw_irs_data/                 # Downloaded ZIP files
│   ├── downloaded_irs_data/          # BMF CSV files
│   ├── extracted_irs_data/           # Extracted CSVs
│   ├── compiled_irs_data/            # Compiled Parquet files
│   └── cleaned_irs_data/             # Final datasets
├── models/                           # ML models and mappings
├── requirements.txt
└── README.md
```

## 📚 Data Outputs

### Compiled Datasets (Parquet format)

Located in `data/compiled_irs_data/`:

- **organization_details_compiled.parquet**: Master list of all organizations with financial data
- **grant_details_compiled.parquet**: All grants made by foundations
- **independent_contractor_details_compiled.parquet**: Contractor compensation data
- **program_related_investments_compiled.parquet**: PRI details
- **direct_charitable_activity_details_compiled.parquet**: Program activities

### Cleaned Datasets

Located in `data/cleaned_irs_data/`:

- **grant_details_recipients_inferred.parquet**: Grants with matched recipient EINs

## 🔧 Configuration

### Extractor Options

```bash
# Force reprocessing of existing files
python -m src.extractors.grant_details_extractor --force-process

# Store all filing data (including null records)
python -m src.extractors.grant_details_extractor --store-all-filing-data

# Parallel processing
python -m src.extractors.grant_details_extractor --max-workers 8
```

### ML Model Options

```bash
# Train with custom hyperparameters
python grant_ttsn_recommender.py
# Edit EMBEDDING_DIM, BATCH_SIZE, NUM_EPOCHS in the script

# Inference with specific model
python grant_ttsn_inference_D2R.py \
  --donor_ein 123456789 \
  --top_k 20 \
  --model_path models/model_epoch_10.pt \
  --embedding_dim 64
```

## 🤖 Machine Learning Model

The project includes a **Two-Tower Neural Network** for matching donors with recipients:

### Architecture
- Separate embedding towers for donors and receivers
- Dot product similarity for matching
- Trained on 4M+ grant records
- 160K+ unique donors, 850K+ unique receivers

### Features
- **Donor → Recipient**: Find potential grant recipients for a foundation
- **Recipient → Donor**: Find potential donors for a nonprofit
- **Batch Processing**: Get recommendations for multiple entities at once
- **Model Checkpoints**: Saved every 5 epochs for experimentation

### Example Usage

```python
from grant_ttsn_inference_D2R import load_model, get_recommendations

# Load trained model
model, mappings = load_model()

# Get top 10 recommendations for a donor
recommendations = get_recommendations(
    donor_ein=123456789,
    model=model,
    mappings=mappings,
    top_k=10
)

# Display results
for rank, (receiver_ein, score) in enumerate(recommendations, 1):
    print(f"{rank}. Receiver EIN {receiver_ein}: {score:.4f}")
```

## 📊 Data Sources

- **IRS Form 990 E-Files**: [IRS.gov Form 990 Downloads](https://www.irs.gov/charities-non-profits/form-990-series-downloads)
- **IRS Business Master File**: [IRS.gov Exempt Organizations](https://www.irs.gov/charities-non-profits/exempt-organizations-business-master-file-extract-eo-bmf)

## 🛠️ Advanced Features

### Fuzzy Matching

The grant matching system uses:
- **RapidFuzz** for high-performance string matching
- **Multi-stage matching**: ZIP code → State-wide → Fallback
- **Parallel processing**: Utilizes all CPU cores
- **Name standardization**: Handles legal entity suffixes

### Performance Optimizations

- **Parallel ZIP processing**: Process multiple ZIP files simultaneously
- **Chunked reading**: Memory-efficient processing of large files
- **Incremental compilation**: Skip already-processed files
- **Parquet format**: Fast I/O with columnar storage

## 📈 Statistics

Based on recent IRS data:
- **163,000+** grant-making foundations
- **847,000+** grant recipients
- **4.1M+** individual grant records
- **$XXX billion** in total grants analyzed

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional data extractors (Schedule A, Schedule D, etc.)
- Enhanced ML models (incorporate text embeddings, financial features)
- Data quality improvements
- Documentation and examples

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- IRS for making 990 data publicly available
- Nonprofit Open Data Collective for research and documentation
- Open-source libraries: pandas, PyTorch, scikit-learn, RapidFuzz

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is for research and educational purposes. Always verify data quality and consult original IRS filings for official information.
