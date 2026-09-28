# CustomerIQ — Snowflake-Free Edition

This edition requires **no Snowflake account or connection**. It supports CSV, Excel, and SQLite data sources and keeps the ML pipeline independent from the storage layer.

## Windows setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If activation is blocked, skip activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or run `run_windows.ps1`.

## Data sources

### CSV
Upload `.csv` from the sidebar.

### Excel
Upload `.xlsx`/`.xls` and choose a sheet.

### SQLite
Upload `.db`, `.sqlite`, or `.sqlite3`, choose a table, or supply a read-only SQL query.

The supplied hackathon CSV remains in `data/Customer_Segmentation_Cleaned_Encoded-1.csv`.

## Pipeline

`CSV/Excel/SQLite → cleaning → RFM/behavioral features → log transform → robust scaling → K-Means → personas → Streamlit`

No Snowflake is required.
