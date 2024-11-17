import sqlite3
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GoalInfo:
    name: str
    minute: int
    offset: Optional[int] = None
    penalty: bool = False
    owngoal: bool = False


class EuroDatabase:
    def __init__(self, db_name=r"C:\Users\kyleh\Desktop\MatchMind_Master\MatchMind_db.db"):
        self.db_name = db_name
        self.setup_logging()
        self.euro_competitions = {
            "2020": "Euro 2020",
            "2024": "Euro 2024"
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f'euro_sync_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


    def fetch_data(self, year: str) -> Optional[Dict]:
        """Fetch Euro championship data for a specific year"""
        try:
            url = f"https://raw.githubusercontent.com/openfootball/euro.json/master/{year}/euro.json"
            self.logger.info(f"Fetching Euro {year} data from: {url}")

            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch Euro {year} data: {str(e)}")
            return None

    def process_goals(self, goals_data: List[Dict]) -> List[GoalInfo]:
        """Process goals data into structured format"""
        goals = []
        for goal in goals_data:
            goals.append(GoalInfo(
                name=goal.get('name', 'Unknown'),
                minute=goal.get('minute', 0),
                offset=goal.get('offset'),
                penalty=goal.get('penalty', False),
                owngoal=goal.get('owngoal', False)
            ))
        return goals

    def process_json_data(self, json_data: Dict, source_file: str) -> List[Dict]:
        """Process Euro JSON data and extract match information"""
        matches = []

        try:
            competition = json_data.get("name", "Unknown Euro Competition")
            season = competition.split()[-1] if competition else "Unknown Season"

            for round_data in json_data.get("rounds", []):
                round_name = round_data.get("name", "Unknown Round")

                for match in round_data.get("matches", []):
                    # Extract basic match information
                    team1_data = match.get("team1", {})
                    team2_data = match.get("team2", {})
                    score_data = match.get("score", {})

                    # Process scores
                    ft_score = score_data.get("ft", [None, None])
                    ht_score = score_data.get("ht", [None, None])

                    # Process goals
                    goals1 = self.process_goals(match.get("goals1", []))
                    goals2 = self.process_goals(match.get("goals2", []))

                    # Calculate additional statistics
                    total_goals = sum(ft_score) if all(s is not None for s in ft_score) else 0
                    ht_total = sum(ht_score) if all(s is not None for s in ht_score) else 0
                    goals_per_half = total_goals - ht_total

                    penalties = sum(1 for g in goals1 + goals2 if g.penalty)
                    own_goals = sum(1 for g in goals1 + goals2 if g.owngoal)

                    match_data = {
                        "match_number": match.get("num"),
                        "competition": competition,
                        "season": season,
                        "round": round_name,
                        "date": match.get("date"),
                        "time": match.get("time"),
                        "team1": team1_data.get("name"),
                        "team2": team2_data.get("name"),
                        "team1_code": team1_data.get("code"),
                        "team2_code": team2_data.get("code"),
                        "group_name": match.get("group"),

                        # Scores
                        "team1_score_ht": ht_score[0],
                        "team2_score_ht": ht_score[1],
                        "team1_score_ft": ft_score[0],
                        "team2_score_ft": ft_score[1],

                        # Goals details
                        "goals_team1": str([vars(g) for g in goals1]),
                        "goals_team2": str([vars(g) for g in goals2]),

                        # Statistics
                        "total_goals": total_goals,
                        "goals_per_half": goals_per_half,
                        "penalties_in_match": penalties,
                        "own_goals_in_match": own_goals,

                        # Additional fields
                        "competition_type": "International Championship",
                        "country": "Europe",
                        "source_file": source_file
                    }

                    # Add calculated fields
                    match_data.update(self.calculate_match_stats(
                        match_data["team1_score_ft"],
                        match_data["team2_score_ft"],
                        match_data["team1_score_ht"],
                        match_data["team2_score_ht"]
                    ))

                    matches.append(match_data)

        except Exception as e:
            self.logger.error(f"Error processing Euro JSON data: {str(e)}")

        return matches

    def calculate_match_stats(self, ft1, ft2, ht1, ht2) -> Dict:
        """Calculate additional match statistics"""
        if None in (ft1, ft2, ht1, ht2):
            return {
                "ht_result": "PENDING",
                "ft_result": "PENDING",
                "score_changed": False,
                "match_status": "SCHEDULED",
                "over_0_5": 0, "over_1_5": 0, "over_2_5": 0,
                "over_3_5": 0, "over_4_5": 0, "over_5_5": 0,
                "under_0_5": 0, "under_1_5": 0, "under_2_5": 0,
                "under_3_5": 0, "under_4_5": 0, "under_5_5": 0,
                "goal_difference": 0,
                "clean_sheet_team1": 0,
                "clean_sheet_team2": 0,
                "btts": 0
            }

        total_goals = ft1 + ft2
        ht_result = "WIN" if ht1 > ht2 else "DRAW" if ht1 == ht2 else "LOSS"
        ft_result = "WIN" if ft1 > ft2 else "DRAW" if ft1 == ft2 else "LOSS"

        return {
            "ht_result": ht_result,
            "ft_result": ft_result,
            "score_changed": ht_result != ft_result,
            "match_status": "COMPLETED",
            "over_0_5": 1 if total_goals > 0.5 else 0,
            "over_1_5": 1 if total_goals > 1.5 else 0,
            "over_2_5": 1 if total_goals > 2.5 else 0,
            "over_3_5": 1 if total_goals > 3.5 else 0,
            "over_4_5": 1 if total_goals > 4.5 else 0,
            "over_5_5": 1 if total_goals > 5.5 else 0,
            "under_0_5": 1 if total_goals < 0.5 else 0,
            "under_1_5": 1 if total_goals < 1.5 else 0,
            "under_2_5": 1 if total_goals < 2.5 else 0,
            "under_3_5": 1 if total_goals < 3.5 else 0,
            "under_4_5": 1 if total_goals < 4.5 else 0,
            "under_5_5": 1 if total_goals < 5.5 else 0,
            "goal_difference": abs(ft1 - ft2),
            "clean_sheet_team1": 1 if ft2 == 0 else 0,
            "clean_sheet_team2": 1 if ft1 == 0 else 0,
            "btts": 1 if ft1 > 0 and ft2 > 0 else 0
        }

    def sync_euro_data(self):
        """Sync all Euro championship data"""
        for year in self.euro_competitions.keys():
            json_data = self.fetch_data(year)
            if json_data:
                matches = self.process_json_data(json_data, f"euro_{year}.json")
                self.insert_data(matches)
                self.logger.info(f"Processed {len(matches)} matches for Euro {year}")

    def initialize_tables(self):
        """Create tables for Euro matches with additional fields"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS euro_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_number INTEGER,
                competition TEXT,
                season TEXT,
                round TEXT,
                date DATE,
                time TEXT,
                team1 TEXT,
                team2 TEXT,
                team1_code TEXT,
                team2_code TEXT,
                group_name TEXT,

                -- Scores
                team1_score_ht INTEGER,
                team2_score_ht INTEGER,
                team1_score_ft INTEGER,
                team2_score_ft INTEGER,

                -- Goals details (stored as JSON)
                goals_team1 TEXT,
                goals_team2 TEXT,

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
                penalties_in_match INTEGER,
                own_goals_in_match INTEGER,

                -- Metadata
                competition_type TEXT,
                country TEXT,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Add unique constraint for match identification
                UNIQUE(competition, season, date, team1, team2)
            )''')

            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_euro_matches_date ON euro_matches(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_euro_matches_teams ON euro_matches(team1, team2)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_euro_matches_competition ON euro_matches(competition, season)')

            conn.commit()
            self.logger.info("Euro tables initialized successfully!")
        except Exception as e:
            self.logger.error(f"Failed to initialize Euro tables: {str(e)}")
        finally:
            conn.close()

    def insert_data(self, matches: List[Dict]):
        """Insert Euro match data into the database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            for match in matches:
                # Get column names from the match dictionary
                columns = match.keys()
                placeholders = ', '.join(['?' for _ in columns])
                values = [match[col] for col in columns]

                # Insert with ON CONFLICT for the unique constraint
                query = f'''
                INSERT INTO euro_matches ({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(competition, season, date, team1, team2) 
                DO UPDATE SET
                {', '.join(f'{col} = excluded.{col}' for col in columns
                           if col not in ['competition', 'season', 'date', 'team1', 'team2'])}
                '''

                cursor.execute(query, values)

            conn.commit()
            self.logger.info(f"Successfully inserted/updated {len(matches)} matches")
        except Exception as e:
            self.logger.error(f"Failed to insert Euro data: {str(e)}")
            # Add more detailed error information
            self.logger.error(f"Error details: {str(e.__cause__)}")
        finally:
            conn.close()


def main():
    try:
        db = EuroDatabase()
        db.initialize_tables()
        db.sync_euro_data()
    except Exception as e:
        print(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()