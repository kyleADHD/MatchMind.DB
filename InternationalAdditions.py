import sqlite3
import logging
import requests
from datetime import datetime


class WorldCupDatabase:
    def __init__(self, db_name=r"C:\Users\kyleh\Desktop\MatchMind_Master\MatchMind_db.db"):
        self.db_name = db_name
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def initialize_tables(self):
        """Create tables for matches"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS world_cup_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                goals_per_half INTEGER,
                ht_result TEXT,
                ft_result TEXT,
                score_changed BOOLEAN,

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
                btts INTEGER,

                -- Metadata
                competition_type TEXT,
                country TEXT,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            conn.commit()
            self.logger.info("Tables initialized successfully!")
        except Exception as e:
            self.logger.error(f"Failed to initialize tables: {str(e)}")
        finally:
            conn.close()

    def fetch_data(self, url):
        """Fetch JSON data from URL"""
        try:
            self.logger.info(f"Fetching data from: {url}")
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch data from {url}: {str(e)}")
            return None

    def process_json_data(self, json_data, source_file):
        """Process JSON data and extract match information"""
        matches = []

        try:
            competition = json_data.get("name", "Unknown Competition")
            season = competition.split()[-1] if competition else "Unknown Season"
            rounds = json_data.get("rounds", [])

            # Define expected columns for the database
            expected_columns = [
                "competition", "season", "round", "date", "time", "team1", "team2",
                "team1_score_ht", "team2_score_ht", "team1_score_ft", "team2_score_ft",
                "total_goals", "goals_per_half", "ht_result", "ft_result", "score_changed",
                "match_status", "over_0_5", "over_1_5", "over_2_5", "over_3_5", "over_4_5", "over_5_5",
                "under_0_5", "under_1_5", "under_2_5", "under_3_5", "under_4_5", "under_5_5",
                "goal_difference", "clean_sheet_team1", "clean_sheet_team2", "btts",
                "competition_type", "country", "source_file"
            ]

            for round_data in rounds:
                round_name = round_data.get("name", "Unknown Round")
                for match in round_data.get("matches", []):
                    # Extract teams and scores
                    team1 = match.get("team1", {}).get("name", "Unknown Team 1")
                    team2 = match.get("team2", {}).get("name", "Unknown Team 2")
                    team1_score_ft = match.get("score1", None)
                    team2_score_ft = match.get("score2", None)
                    team1_score_ht = match.get("score1i", None)
                    team2_score_ht = match.get("score2i", None)

                    # Calculate basic statistics
                    total_goals = (team1_score_ft or 0) + (team2_score_ft or 0)
                    goals_per_half = total_goals - ((team1_score_ht or 0) + (team2_score_ht or 0))
                    ht_result = (
                        "DRAW" if team1_score_ht == team2_score_ht else
                        ("WIN" if team1_score_ht > team2_score_ht else "LOSE")
                    ) if team1_score_ht is not None and team2_score_ht is not None else "UNKNOWN"
                    ft_result = (
                        "DRAW" if team1_score_ft == team2_score_ft else
                        ("WIN" if team1_score_ft > team2_score_ft else "LOSE")
                    )
                    score_changed = ht_result != ft_result

                    # Calculate over/under statistics
                    over_under_metrics = {
                        f"over_{x}_5": 1 if total_goals > x else 0 for x in range(6)
                    }
                    over_under_metrics.update({
                        f"under_{x}_5": 1 if total_goals <= x else 0 for x in range(6)
                    })

                    # Ensure all expected keys are present
                    expected_metrics = [f"over_{x}_5" for x in range(6)] + [f"under_{x}_5" for x in range(6)]
                    for key in expected_metrics:
                        if key not in over_under_metrics:
                            over_under_metrics[key] = 0

                    # Additional statistics
                    goal_difference = abs((team1_score_ft or 0) - (team2_score_ft or 0))
                    clean_sheet_team1 = 1 if team2_score_ft == 0 else 0
                    clean_sheet_team2 = 1 if team1_score_ft == 0 else 0
                    btts = 1 if team1_score_ft > 0 and team2_score_ft > 0 else 0

                    # Build match data
                    match_data = {
                        "competition": competition,
                        "season": season,
                        "round": round_name,
                        "date": match.get("date"),
                        "time": match.get("time"),
                        "team1": team1,
                        "team2": team2,
                        "team1_score_ht": team1_score_ht,
                        "team2_score_ht": team2_score_ht,
                        "team1_score_ft": team1_score_ft,
                        "team2_score_ft": team2_score_ft,
                        "total_goals": total_goals,
                        "goals_per_half": goals_per_half,
                        "ht_result": ht_result,
                        "ft_result": ft_result,
                        "score_changed": score_changed,
                        "match_status": "COMPLETED",
                        **over_under_metrics,
                        "goal_difference": goal_difference,
                        "clean_sheet_team1": clean_sheet_team1,
                        "clean_sheet_team2": clean_sheet_team2,
                        "btts": btts,
                        "competition_type": "Cup",
                        "country": "International",
                        "source_file": source_file
                    }

                    # Trim match_data to match expected columns
                    match_data = {key: match_data[key] for key in expected_columns if key in match_data}

                    matches.append(match_data)
        except Exception as e:
            self.logger.error(f"Error processing JSON data: {str(e)}")

        return matches

    def insert_data(self, matches):
        """Insert match data into the database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Define the exact columns to insert, excluding auto-increment or default columns
            columns = [
                "competition", "season", "round", "date", "time", "team1", "team2",
                "team1_score_ht", "team2_score_ht", "team1_score_ft", "team2_score_ft",
                "total_goals", "goals_per_half", "ht_result", "ft_result", "score_changed",
                "match_status", "over_0_5", "over_1_5", "over_2_5", "over_3_5", "over_4_5", "over_5_5",
                "under_0_5", "under_1_5", "under_2_5", "under_3_5", "under_4_5", "under_5_5",
                "goal_difference", "clean_sheet_team1", "clean_sheet_team2", "btts",
                "competition_type", "country", "source_file"
            ]

            for match in matches:
                # Map match data to the defined columns
                values = tuple(match[col] for col in columns)

                cursor.execute(f'''
                INSERT INTO world_cup_matches ({", ".join(columns)}) 
                VALUES ({", ".join(["?"] * len(columns))})
                ''', values)

            conn.commit()
            self.logger.info("Data inserted successfully!")
        except Exception as e:
            self.logger.error(f"Failed to insert data: {str(e)}")
        finally:
            conn.close()


if __name__ == "__main__":
    db = WorldCupDatabase()
    db.initialize_tables()

    # Fetch and process JSON data
    url = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json"
    source_file = "2014_worldcup.json"
    json_data = db.fetch_data(url)

    if json_data:
        matches = db.process_json_data(json_data, source_file)
        db.insert_data(matches)
