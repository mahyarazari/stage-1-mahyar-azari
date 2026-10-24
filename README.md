# NYC 311 Data Factory — Stage 1

## Getting Started

From the project root directory, run the following commands using **Python 3.11 or later**:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# For macOS/Linux, use:
# source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Install the project in editable mode
python -m pip install -e .

# Run all tests (no internet connection required)
python -m unittest discover -s tests -t .

# Execute the complete data pipeline
data-factory run
```

## Pipeline Execution

The data factory follows a five-stage pipeline:

**Fetch → Validate → Clean → Enrich → Publish**

Additional commands:

```bash
# Run specific pipeline stages
data-factory run --stages validate,clean

# Process only new data since the previous run
data-factory run --since-last-run

# Validate project configuration
data-factory check-config
```

## Project Structure and Deliverables

| File / Directory | Description |
|---|---|
| `factory.toml` | Pipeline configuration |
| `data/curated/` | Processed datasets and pipeline outputs |
| `run_manifest.json` | Execution metadata, stored in `data/curated/` |
| `solution.ipynb` | Main analysis notebook; execute using **Restart & Run All** |
| `report.md` | Summary of findings and implementation decisions |
| `docs/decisions.md` | Technical decisions and their rationale |

## Exit Codes

The CLI uses the following exit codes to indicate execution status:

| Exit Code | Description |
|---|---|
| `0` | Successful execution |
| `10` | Configuration error |
| `20–21` | Network-related errors |
| `30–33` | Data-related errors |
| `40` | Missing input from a preceding pipeline stage |
| `1` | Unexpected or unhandled error |

All unit tests can be executed **offline**, without an internet connection.
