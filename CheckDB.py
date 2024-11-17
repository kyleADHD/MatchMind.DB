import sqlite3
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from collections import defaultdict


class DatabaseInspector:
    def __init__(self, db_path: str = 'MatchMind_db.db'):
        self.db_path = db_path
        self.setup_logging()



    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f'db_inspection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_table_info(self) -> Dict[str, List[Dict]]:
        """Get detailed information about all tables in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            table_info = {}
            for table in tables:
                table_name = table[0]
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                # Get index info
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()

                # Get foreign key info
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = cursor.fetchall()

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                table_info[table_name] = {
                    'columns': [
                        {
                            'cid': col[0],
                            'name': col[1],
                            'type': col[2],
                            'notnull': col[3],
                            'default': col[4],
                            'pk': col[5]
                        } for col in columns
                    ],
                    'indexes': [
                        {
                            'name': idx[1],
                            'unique': idx[2]
                        } for idx in indexes
                    ],
                    'foreign_keys': [
                        {
                            'id': fk[0],
                            'seq': fk[1],
                            'table': fk[2],
                            'from': fk[3],
                            'to': fk[4]
                        } for fk in foreign_keys
                    ],
                    'row_count': row_count
                }

            return table_info

        except Exception as e:
            self.logger.error(f"Error getting table info: {str(e)}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()

    def get_table_statistics(self, table_name: str) -> Dict:
        """Get detailed statistics for a specific table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]

            stats = {
                'row_count': 0,
                'null_counts': {},
                'distinct_counts': {},
                'min_values': {},
                'max_values': {},
                'sample_values': {}
            }

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            stats['row_count'] = cursor.fetchone()[0]

            # Get statistics for each column
            for col in columns:
                # Null count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
                stats['null_counts'][col] = cursor.fetchone()[0]

                # Distinct count
                cursor.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table_name}")
                stats['distinct_counts'][col] = cursor.fetchone()[0]

                # Min and Max values (for numeric and date columns)
                try:
                    cursor.execute(f"SELECT MIN({col}), MAX({col}) FROM {table_name}")
                    min_val, max_val = cursor.fetchone()
                    stats['min_values'][col] = min_val
                    stats['max_values'][col] = max_val
                except:
                    stats['min_values'][col] = None
                    stats['max_values'][col] = None

                # Sample values
                cursor.execute(f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT 5")
                stats['sample_values'][col] = [row[0] for row in cursor.fetchall()]

            return stats

        except Exception as e:
            self.logger.error(f"Error getting statistics for table {table_name}: {str(e)}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()

    def check_data_integrity(self) -> Dict:
        """Check data integrity across tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            integrity_issues = {
                'null_primary_keys': defaultdict(int),
                'duplicate_unique_values': defaultdict(list),
                'invalid_dates': defaultdict(list),
                'inconsistent_foreign_keys': defaultdict(list)
            }

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]

                # Check primary keys
                cursor.execute(f"PRAGMA table_info({table_name})")
                pk_columns = [col[1] for col in cursor.fetchall() if col[5]]  # col[5] is pk flag

                for pk in pk_columns:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {pk} IS NULL")
                    null_count = cursor.fetchone()[0]
                    if null_count > 0:
                        integrity_issues['null_primary_keys'][f"{table_name}.{pk}"] = null_count

                # Check unique constraints
                cursor.execute(f"PRAGMA index_list({table_name})")
                for index in cursor.fetchall():
                    if index[2]:  # is unique
                        cursor.execute(f"PRAGMA index_info({index[1]})")
                        columns = [info[2] for info in cursor.fetchall()]
                        column_list = ', '.join(columns)

                        cursor.execute(f"""
                            SELECT {column_list}, COUNT(*)
                            FROM {table_name}
                            GROUP BY {column_list}
                            HAVING COUNT(*) > 1
                        """)
                        duplicates = cursor.fetchall()
                        if duplicates:
                            integrity_issues['duplicate_unique_values'][f"{table_name}.{index[1]}"] = duplicates

                # Check date formats
                cursor.execute(f"PRAGMA table_info({table_name})")
                date_columns = [col[1] for col in cursor.fetchall() if 'date' in col[2].lower()]

                for date_col in date_columns:
                    cursor.execute(f"""
                        SELECT {date_col}
                        FROM {table_name}
                        WHERE {date_col} IS NOT NULL
                        AND {date_col} NOT LIKE '____-__-__'
                    """)
                    invalid_dates = cursor.fetchall()
                    if invalid_dates:
                        integrity_issues['invalid_dates'][f"{table_name}.{date_col}"] = invalid_dates

            return integrity_issues

        except Exception as e:
            self.logger.error(f"Error checking data integrity: {str(e)}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()

    def print_schema_report(self):
        """Print comprehensive schema report"""
        print("\n=== Database Schema Report ===")
        print(f"Database: {Path(self.db_path).absolute()}")
        print(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

        table_info = self.get_table_info()

        for table_name, info in table_info.items():
            print(f"\nTable: {table_name}")
            print("-" * 50)

            # Print columns
            print("\nColumns:")
            columns_data = [
                [col['name'], col['type'],
                 'NOT NULL' if col['notnull'] else 'NULL',
                 'PK' if col['pk'] else '',
                 col['default'] if col['default'] else '']
                for col in info['columns']
            ]
            print(tabulate(columns_data,
                           headers=['Name', 'Type', 'Null', 'Key', 'Default'],
                           tablefmt='pipe'))

            # Print indexes
            if info['indexes']:
                print("\nIndexes:")
                for idx in info['indexes']:
                    print(f"- {idx['name']} {'(UNIQUE)' if idx['unique'] else ''}")

            # Print foreign keys
            if info['foreign_keys']:
                print("\nForeign Keys:")
                for fk in info['foreign_keys']:
                    print(f"- {fk['from']} -> {fk['table']}.{fk['to']}")

            # Print basic statistics
            print(f"\nRow count: {info['row_count']:,}")

            # Get and print detailed statistics
            stats = self.get_table_statistics(table_name)
            if stats:
                print("\nColumn Statistics:")
                stats_data = []
                for col in info['columns']:
                    col_name = col['name']
                    stats_data.append([
                        col_name,
                        stats['null_counts'].get(col_name, 'N/A'),
                        stats['distinct_counts'].get(col_name, 'N/A'),
                        stats['min_values'].get(col_name, 'N/A'),
                        stats['max_values'].get(col_name, 'N/A')
                    ])
                print(tabulate(stats_data,
                               headers=['Column', 'Nulls', 'Distinct', 'Min', 'Max'],
                               tablefmt='pipe'))

        # Print integrity check results
        print("\n=== Data Integrity Check ===")
        integrity_issues = self.check_data_integrity()

        if any(issues for issues in integrity_issues.values()):
            print("\nIssues found:")
            for issue_type, issues in integrity_issues.items():
                if issues:
                    print(f"\n{issue_type.replace('_', ' ').title()}:")
                    for location, details in issues.items():
                        print(f"- {location}: {details}")
        else:
            print("\nNo integrity issues found.")


def main():
    try:
        inspector = DatabaseInspector(r"C:\Users\kyleh\Desktop\MatchMind_Master\MatchMind_db.db")
        inspector.print_schema_report()
    except Exception as e:
        print(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()