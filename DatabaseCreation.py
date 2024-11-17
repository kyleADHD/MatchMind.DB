import sqlite3
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from pathlib import Path


class FootballDatabaseInitializer:
    def __init__(self, db_name: str = 'MatchMind_db.db'):
        self.db_name = db_name
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_database(self) -> bool:
        """Create database with essential tables and columns"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Create matches table with essential and calculable columns
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Basic match information
                competition TEXT,
                season TEXT,
                round TEXT,
                date DATE,
                time TEXT,
                team1 TEXT,
                team2 TEXT,

                -- Scores
                team1_score_ht INTEGER,
                team2_score_ht INTEGER,
                team1_score_ft INTEGER,
                team2_score_ft INTEGER,

                -- Basic calculated statistics
                total_goals INTEGER,
                goals_per_half INTEGER,  -- Goals scored in second half
                ht_result TEXT,          -- Result at half time
                ft_result TEXT,          -- Final result
                score_changed BOOLEAN,    -- If result changed from HT to FT

                -- Match status
                match_status TEXT CHECK(match_status IN ('SCHEDULED', 'COMPLETED', 'POSTPONED', 'CANCELLED')),

                -- Over/under values
                over_0_5 INTEGER,
                over_1_5 INTEGER,
                over_2_5 INTEGER,
                over_3_5 INTEGER,
                over_4_5 INTEGER,
                over_5_5 INTEGER,
                under_0_5 INTEGER,
                under_1_5 INTEGER,
                under_2_5 INTEGER,
                under_3_5 INTEGER,
                under_4_5 INTEGER,
                under_5_5 INTEGER,

                -- Additional match statistics
                goal_difference INTEGER,
                clean_sheet_team1 INTEGER,
                clean_sheet_team2 INTEGER,
                btts INTEGER,            -- Both teams to score

                -- Metadata
                competition_type TEXT,    -- League or Cup
                country TEXT,
                source_file TEXT,        -- Source JSON file
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP
            )''')

            # Create clubs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS clubs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Basic club information
                name TEXT,
                code TEXT,
                country TEXT,
                season TEXT,
                competition TEXT,

                -- Metadata
                source_file TEXT,        -- Source JSON file
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP,

                -- Ensure unique clubs per season/competition
                UNIQUE(name, season, competition)
            )''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team1, team2)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_competition ON matches(competition, season)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_clubs_name ON clubs(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_source ON matches(source_file)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_clubs_source ON clubs(source_file)')

            conn.commit()
            self.logger.info("Database initialized successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()

    def verify_database(self) -> Dict:
        """Verify database structure and setup"""
        results = {
            'database_exists': False,
            'tables_created': False,
            'table_counts': {},
            'indexes_created': False,
            'structure_valid': False
        }

        try:
            db_path = Path(self.db_name)
            results['database_exists'] = db_path.exists()

            if not results['database_exists']:
                self.logger.error("Database file does not exist!")
                return results

            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            required_tables = {'matches', 'clubs'}

            results['tables_created'] = required_tables.issubset(set(tables))

            # Get table counts
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                results['table_counts'][table] = cursor.fetchone()[0]

            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = set(row[0] for row in cursor.fetchall())
            required_indexes = {
                'idx_matches_date',
                'idx_matches_teams',
                'idx_matches_competition',
                'idx_clubs_name',
                'idx_matches_source',
                'idx_clubs_source'
            }

            results['indexes_created'] = required_indexes.issubset(indexes)

            # Print table structures for verification
            print("\nTable Structures:")
            for table in tables:
                print(f"\n{table} table columns:")
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"- {col[1]} ({col[2]})")

            results['structure_valid'] = True
            return results

        except Exception as e:
            self.logger.error(f"Database verification failed: {str(e)}")
            return results
        finally:
            if 'conn' in locals():
                conn.close()


def main():
    initializer = FootballDatabaseInitializer()

    # Initialize database
    if initializer.initialize_database():
        print("Database initialization successful!")
    else:
        print("Database initialization failed!")
        return

    # Verify the setup
    verification_results = initializer.verify_database()

    # Print verification results
    print("\nDatabase Verification Results:")
    print("-" * 30)
    print(f"Database exists: {verification_results['database_exists']}")
    print(f"Tables created: {verification_results['tables_created']}")
    print(f"Indexes created: {verification_results['indexes_created']}")
    print(f"Structure valid: {verification_results['structure_valid']}")
    print("\nTable row counts:")
    for table, count in verification_results['table_counts'].items():
        print(f"- {table}: {count} rows")


if __name__ == "__main__":
    main()