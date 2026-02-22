"""File I/O, database, and network handlers for data ingestion and storage."""

import os
import json
import sqlite3


def load_and_process_csv(filepath, delimiter=","):
    """Load a CSV file and return processed records.

    Category B: File I/O side effect.
    """
    records = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        return records

    headers = lines[0].strip().split(delimiter)
    for line in lines[1:]:
        values = line.strip().split(delimiter)
        record = {}
        for i in range(len(headers)):
            if i < len(values):
                record[headers[i]] = values[i]
            else:
                record[headers[i]] = None
        records.append(record)

    return records


def save_results_to_db(results, db_path, table_name="results"):
    """Save analysis results to a SQLite database.

    Category B: Database side effect.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if not results:
        conn.close()
        return 0

    columns = list(results[0].keys())
    col_defs = ", ".join([f"{col} TEXT" for col in columns])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")

    count = 0
    for result in results:
        placeholders = ", ".join(["?" for _ in columns])
        values = [str(result.get(col, "")) for col in columns]
        cursor.execute(f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})", values)
        count += 1

    conn.commit()
    conn.close()
    return count


def fetch_remote_dataset(url, timeout=30):
    """Fetch a dataset from a remote URL.

    Category B: Network side effect.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "DataToolkit/1.0")
        response = urllib.request.urlopen(req, timeout=timeout)
        data = response.read().decode("utf-8")
        return json.loads(data)
    except urllib.error.URLError as e:
        return {"error": str(e)}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
