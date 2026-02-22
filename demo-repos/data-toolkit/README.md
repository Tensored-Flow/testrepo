# Data Toolkit

A lightweight Python library for data processing, transformation, and analysis.
Built for internal ETL pipelines and data cleaning workflows.

## Modules

- **transforms** — Data deduplication, normalization, flattening
- **search** — Pair matching, lookup operations
- **analytics** — Moving averages, statistical summaries
- **io_handlers** — CSV loading, database storage, remote fetching
- **api_client** — Internal data API integration

## Usage

```python
from toolkit.transforms import deduplicate_records, normalize_column

# Deduplicate by composite key
records = [{"name": "Alice", "dept": "Eng"}, {"name": "Alice", "dept": "Eng"}]
unique = deduplicate_records(records, ["name", "dept"])

# Normalize values to 0-1
normalized = normalize_column([10, 20, 30, 40, 50])
```

## Running Tests

```bash
python -m pytest tests/
```
