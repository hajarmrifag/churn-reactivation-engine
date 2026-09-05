from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "berka"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "berka.sqlite"

TABLES = {
    "account": "account.csv",
    "card": "card.csv",
    "client": "client.csv",
    "disp": "disp.csv",
    "district": "district.csv",
    "loan": "loan.csv",
    "order_table": "order.csv",
    "trans": "trans.csv",
}


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        for table_name, filename in TABLES.items():
            path = RAW_DIR / filename
            print(f"Loading {filename} -> {table_name}...")

            df = pd.read_csv(path, sep=";", low_memory=False)
            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=False,
                chunksize=50_000,
            )

            count = conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            print(f"  {count:,} rows")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trans_account_date "
            "ON trans(account_id, date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_disp_account "
            "ON disp(account_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_loan_account "
            "ON loan(account_id)"
        )

    print(f"\nSQLite database created: {DB_PATH}")


if __name__ == "__main__":
    main()
