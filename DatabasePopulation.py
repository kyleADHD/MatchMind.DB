import sqlite3
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import os
from dataclasses import dataclass
from enum import Enum


class MatchResult(Enum):
    WIN = 'WIN'
    DRAW = 'DRAW'
    LOSS = 'LOSS'
    PENDING = 'PENDING'


class MatchStatus(Enum):
    SCHEDULED = 'SCHEDULED'
    COMPLETED = 'COMPLETED'
    POSTPONED = 'POSTPONED'
    CANCELLED = 'CANCELLED'


@dataclass
class Score:
    team1: int = 0
    team2: int = 0

    @property
    def total(self) -> int:
        return self.team1 + self.team2

    def get_result(self, perspective_team1: bool = True) -> MatchResult:
        if self.team1 > self.team2:
            return MatchResult.WIN if perspective_team1 else MatchResult.LOSS
        elif self.team1 < self.team2:
            return MatchResult.LOSS if perspective_team1 else MatchResult.WIN
        return MatchResult.DRAW

    def is_clean_sheet(self, for_team1: bool = True) -> bool:
        return self.team2 == 0 if for_team1 else self.team1 == 0

    def is_btts(self) -> bool:
        return self.team1 > 0 and self.team2 > 0


@dataclass
class MatchScores:
    ht: Score = None
    ft: Score = None

    def __post_init__(self):
        self.ht = self.ht or Score()
        self.ft = self.ft or Score()

    @property
    def goals_per_half(self) -> Tuple[int, int]:
        first_half = self.ht.total
        second_half = self.ft.total - first_half
        return first_half, second_half

    @property
    def score_changed(self) -> bool:
        return self.ht.get_result() != self.ft.get_result()


class FootballDataProcessor:
    def __init__(self, db_name: str = r'C:\Users\kyleh\Desktop\MatchMind_Master\football_data.db',
                 repo_path: str = r'C:\Users\kyleh\Desktop\MatchMind_Master\football.json-master'):
        self.db_name = db_name
        self.repo_path = Path(repo_path)
        self.setup_logging()
        self.verify_database_structure()
        self.conn = sqlite3.connect(self.db_name)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.error_log = []
        self.stats = {
            'files_processed': 0,
            'matches_processed': 0,
            'clubs_processed': 0,
            'errors': 0,
            'warnings': 0,
            'leagues_processed': set()
        }

    def process_all(self):
        """Process all files in the repository"""
        self.logger.info("Starting data processing...")

        try:
            # Process club files first
            club_files = list(self.repo_path.rglob('*.clubs.json'))
            self.logger.info(f"Found {len(club_files)} club files")

            # Process clubs with progress bar
            for file_path in tqdm(club_files, desc="Processing club files"):
                try:
                    self.process_club_file(file_path)
                except Exception as e:
                    self.logger.error(f"Error processing club file {file_path}: {str(e)}")
                    self.stats['errors'] += 1
                    continue

            # Process match files (excluding club files)
            match_files = [f for f in self.repo_path.rglob('*.json')
                           if not f.name.endswith('.clubs.json')]
            self.logger.info(f"Found {len(match_files)} match files")

            # Process matches with progress bar
            for file_path in tqdm(match_files, desc="Processing match files"):
                try:
                    self.process_matches_file(file_path)
                except Exception as e:
                    self.logger.error(f"Error processing match file {file_path}: {str(e)}")
                    self.stats['errors'] += 1
                    continue

            # Save error log if any errors occurred
            if self.error_log:
                log_path = Path('processing_errors.log')
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.error_log))
                self.logger.info(f"Error log saved to {log_path}")

            self.logger.info("Processing completed")
            return self.stats

        except Exception as e:
            self.logger.error(f"Fatal error during processing: {str(e)}")
            self.error_log.append(f"Fatal error: {str(e)}")
            raise

    def setup_logging(self):
        """Setup enhanced logging configuration"""
        log_file = f'football_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def verify_database_structure(self):
        """Verify database exists and has correct structure"""
        if not os.path.exists(self.db_name):
            raise FileNotFoundError(f"Database {self.db_name} not found. Please run DatabaseCreation.py first.")

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {table[0] for table in cursor.fetchall()}
            required_tables = {'matches', 'clubs'}

            if not required_tables.issubset(tables):
                missing = required_tables - tables
                raise Exception(f"Missing required tables: {missing}")

            # Verify table structures
            for table in required_tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = {row[1] for row in cursor.fetchall()}

                if table == 'matches':
                    required_columns = {
                        'competition', 'season', 'date', 'team1', 'team2',
                        'team1_score_ft', 'team2_score_ft', 'team1_score_ht', 'team2_score_ht',
                        'match_status', 'source_file', 'last_updated'
                    }
                else:  # clubs table
                    required_columns = {
                        'name', 'code', 'country', 'season', 'competition',
                        'source_file', 'last_updated'
                    }

                missing_columns = required_columns - columns
                if missing_columns:
                    raise Exception(f"Missing required columns in {table}: {missing_columns}")

        except Exception as e:
            conn.close()
            raise Exception(f"Database structure verification failed: {str(e)}")

        conn.close()

    def process_matches_file(self, file_path: Path) -> None:
        """Process matches data file with support for both old and new formats"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract file info
            season = str(file_path.parent.name)
            country = self._extract_country_from_path(file_path)
            competition_type = 'Cup' if 'cup' in str(file_path).lower() else 'League'
            source_file = str(file_path.relative_to(self.repo_path))

            # Get competition name
            if 'name' in data:
                competition = data['name'].split(' 20')[0].strip()
            else:
                competition = self._extract_competition_from_filename(file_path)

            # Handle different match structures
            matches = []
            if 'matches' in data:
                matches = data['matches']
            elif 'rounds' in data:
                for round_data in data['rounds']:
                    if 'matches' in round_data:
                        for match in round_data['matches']:
                            match['round'] = round_data.get('name', '')
                            matches.append(match)

            if not matches:
                self.logger.warning(f"No matches found in {file_path}")
                self.stats['warnings'] += 1
                return

            self.stats['leagues_processed'].add(competition)

            # Process matches with progress bar
            desc = f"Processing {competition} {season}"
            successful_matches = 0

            for match in tqdm(matches, desc=desc, leave=False):
                try:
                    # Validate match data first
                    if not self._validate_match_data(match):
                        self.stats['errors'] += 1
                        continue

                    match_data = self.process_match(
                        match,
                        competition=competition,
                        season=season,
                        competition_type=competition_type,
                        country=country,
                        source_file=source_file
                    )

                    if self.insert_match_data(match_data):
                        successful_matches += 1
                    else:
                        self.stats['errors'] += 1

                except Exception as e:
                    self.logger.error(f"Error processing match in {file_path}: {str(e)}")
                    self.error_log.append(f"Match processing error in {source_file}: {str(e)}")
                    self.stats['errors'] += 1

            self.stats['matches_processed'] += successful_matches
            self.stats['files_processed'] += 1

            self.logger.info(
                f"Processed {successful_matches}/{len(matches)} matches from {source_file}"
            )

        except Exception as e:
            self.logger.error(f"Error processing matches file {file_path}: {str(e)}")
            self.error_log.append(f"Match file error: {source_file}: {str(e)}")
            self.stats['errors'] += 1

    def _validate_match_data(self, match: Dict) -> bool:
        """Validate match data before processing"""
        try:
            # Check required fields
            required_fields = {'date', 'team1', 'team2'}
            if not all(field in match for field in required_fields):
                missing = required_fields - set(match.keys())
                self.logger.warning(f"Missing required fields: {missing}")
                return False

            # Validate date format
            try:
                datetime.strptime(match['date'], '%Y-%m-%d')
            except ValueError:
                self.logger.warning(f"Invalid date format: {match['date']}")
                return False

            # Validate team names
            if not match['team1'].strip() or not match['team2'].strip():
                self.logger.warning("Empty team name")
                return False

            # Validate scores if present
            if 'score' in match:
                if not self._validate_score_data(match['score']):
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"Match validation error: {str(e)}")
            return False

    def _validate_score_data(self, score_data: Dict) -> bool:
        """Validate score data structure"""
        try:
            if isinstance(score_data, dict):
                # Check full time score
                if 'ft' in score_data:
                    ft_score = score_data['ft']
                    if not (isinstance(ft_score, list) and len(ft_score) == 2 and
                            all(isinstance(x, (int, float)) for x in ft_score)):
                        self.logger.warning("Invalid full time score format")
                        return False

                # Check half time score if present
                if 'ht' in score_data:
                    ht_score = score_data['ht']
                    if not (isinstance(ht_score, list) and len(ht_score) == 2 and
                            all(isinstance(x, (int, float)) for x in ht_score)):
                        self.logger.warning("Invalid half time score format")
                        return False

                    # Validate half time vs full time scores
                    if 'ft' in score_data:
                        ft_score = score_data['ft']
                        if not (ht_score[0] <= ft_score[0] and ht_score[1] <= ft_score[1]):
                            self.logger.warning("Half time scores greater than full time scores")
                            return False

            return True

        except Exception as e:
            self.logger.warning(f"Score validation error: {str(e)}")
            return False

    def process_match(self, match: Dict, competition: str, season: str,
                      competition_type: str, country: str, source_file: str) -> Dict:
        """Process individual match data with better error handling"""
        try:
            match_date = datetime.strptime(match.get('date', ''), '%Y-%m-%d')
            is_future = match_date > datetime.now()

            # Create base match data with safe string handling
            match_data = {
                'competition': str(competition).strip(),
                'season': str(season).strip(),
                'round': str(match.get('round', '')).strip(),
                'date': match.get('date', ''),
                'time': str(match.get('time', '')).strip(),
                'team1': str(match.get('team1', '')).strip(),
                'team2': str(match.get('team2', '')).strip(),
                'competition_type': str(competition_type).strip(),
                'country': str(country).strip(),
                'source_file': str(source_file).strip(),
                'match_status': MatchStatus.SCHEDULED.value if is_future else MatchStatus.COMPLETED.value,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Process scores for completed matches
            if not is_future and 'score' in match:
                try:
                    scores = self._process_score(match['score'])

                    # Basic scores with safe integer conversion
                    match_data.update({
                        'team1_score_ft': int(scores.ft.team1),
                        'team2_score_ft': int(scores.ft.team2),
                        'team1_score_ht': int(scores.ht.team1),
                        'team2_score_ht': int(scores.ht.team2),
                        'total_goals': int(scores.ft.total),
                        'goals_per_half': int(scores.goals_per_half[1]),  # Second half goals
                        'ht_result': scores.ht.get_result().value,
                        'ft_result': scores.ft.get_result().value,
                        'score_changed': int(scores.score_changed),
                        'btts': int(scores.ft.is_btts()),
                        'clean_sheet_team1': int(scores.ft.is_clean_sheet(True)),
                        'clean_sheet_team2': int(scores.ft.is_clean_sheet(False)),
                        'goal_difference': int(scores.ft.team1 - scores.ft.team2)
                    })

                    # Calculate over/under values
                    total_goals = scores.ft.total
                    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                        threshold_str = str(threshold).replace('.', '_')
                        match_data[f'over_{threshold_str}'] = int(total_goals > threshold)
                        match_data[f'under_{threshold_str}'] = int(total_goals < threshold)

                except Exception as e:
                    self.logger.error(f"Error processing scores: {str(e)}")
                    # Set default values for score-related fields
                    match_data.update({
                        'team1_score_ft': 0,
                        'team2_score_ft': 0,
                        'team1_score_ht': 0,
                        'team2_score_ht': 0,
                        'total_goals': 0,
                        'goals_per_half': 0,
                        'ht_result': MatchResult.PENDING.value,
                        'ft_result': MatchResult.PENDING.value,
                        'score_changed': 0,
                        'btts': 0,
                        'clean_sheet_team1': 0,
                        'clean_sheet_team2': 0,
                        'goal_difference': 0
                    })
                    # Set default over/under values
                    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                        threshold_str = str(threshold).replace('.', '_')
                        match_data[f'over_{threshold_str}'] = 0
                        match_data[f'under_{threshold_str}'] = 0

            return match_data

        except Exception as e:
            self.logger.error(f"Error in process_match: {str(e)}")
            raise

    def _process_score(self, score_data: Dict) -> MatchScores:
        """Process match score data with better error handling"""
        try:
            if not score_data or not isinstance(score_data, dict):
                return MatchScores()

            ft_score = Score()
            ht_score = Score()

            # Process full time score
            if 'ft' in score_data and isinstance(score_data['ft'], (list, tuple)):
                try:
                    ft_score = Score(
                        team1=int(score_data['ft'][0]),
                        team2=int(score_data['ft'][1])
                    )
                except (IndexError, ValueError, TypeError):
                    self.logger.warning("Invalid full time score format")

            elif 'et' in score_data and isinstance(score_data['et'], (list, tuple)):
                try:
                    ft_score = Score(
                        team1=int(score_data['et'][0]),
                        team2=int(score_data['et'][1])
                    )
                except (IndexError, ValueError, TypeError):
                    self.logger.warning("Invalid extra time score format")

            # Process half time score
            if 'ht' in score_data and isinstance(score_data['ht'], (list, tuple)):
                try:
                    ht_score = Score(
                        team1=int(score_data['ht'][0]),
                        team2=int(score_data['ht'][1])
                    )
                except (IndexError, ValueError, TypeError):
                    self.logger.warning("Invalid half time score format")

            # Validate half time scores are not greater than full time
            if (ht_score.team1 > ft_score.team1 or
                    ht_score.team2 > ft_score.team2):
                self.logger.warning("Half time scores greater than full time scores")
                ht_score = Score()  # Reset to default if invalid

            return MatchScores(ht=ht_score, ft=ft_score)

        except Exception as e:
            self.logger.error(f"Error processing scores: {str(e)}")
            return MatchScores()

    def insert_match_data(self, match_data: Dict) -> bool:
        """Insert match data with better error handling and validation"""
        try:
            cursor = self.conn.cursor()

            # Validate required fields
            required_fields = {'competition', 'season', 'date', 'team1', 'team2', 'match_status'}
            if not all(field in match_data for field in required_fields):
                missing = required_fields - set(match_data.keys())
                self.logger.error(f"Missing required fields: {missing}")
                return False

            # Ensure all values are the correct type
            for key, value in match_data.items():
                if value is None:
                    if key.endswith(('_ft', '_ht', 'total_goals', '_streak', '_count', 'btts')):
                        match_data[key] = 0
                    elif key.endswith(('_score', '_probability')):
                        match_data[key] = 0.0
                    else:
                        match_data[key] = ''

            columns = list(match_data.keys())
            values = [match_data[col] for col in columns]
            placeholders = ','.join(['?' for _ in columns])

            sql = f'''
            INSERT OR REPLACE INTO matches (
                {','.join(columns)}
            ) VALUES ({placeholders})
            '''

            cursor.execute(sql, values)
            self.conn.commit()
            return True

        except Exception as e:
            self.logger.error(f"Error inserting match data: {str(e)}")
            self.error_log.append(f"Match insertion error: {str(e)}")
            return False

    def process_club_file(self, file_path: Path) -> None:
        """Process club information from JSON file"""
        try:
            source_file = str(file_path.relative_to(self.repo_path))

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract file info
            if 'name' in data:
                competition = data['name'].split(' 20')[0].strip()
            else:
                competition = self._extract_competition_from_filename(file_path)

            season = str(file_path.parent.name)
            country = self._extract_country_from_path(file_path)

            if 'clubs' not in data or not data['clubs']:
                self.logger.warning(f"No clubs found in {source_file}")
                self.stats['warnings'] += 1
                return

            successful_clubs = 0
            for club in data['clubs']:
                try:
                    if not self._validate_club_data(club):
                        continue

                    club_data = {
                        'name': str(club.get('name', '')).strip() if club.get('name') else 'Unknown',
                        'code': str(club.get('code', '')).strip() if club.get('code') else '',
                        'country': str(club.get('country', country)).strip(),
                        'season': season,
                        'competition': competition,
                        'source_file': source_file,
                        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    if self._insert_club_data(club_data):
                        successful_clubs += 1
                    else:
                        self.stats['errors'] += 1

                except Exception as e:
                    self.logger.error(f"Error processing club {club.get('name', 'Unknown')} in {source_file}: {str(e)}")
                    self.error_log.append(f"Club processing error in {source_file}: {str(e)}")
                    self.stats['errors'] += 1

            self.stats['clubs_processed'] += successful_clubs
            self.stats['files_processed'] += 1

            self.logger.info(
                f"Processed {successful_clubs}/{len(data['clubs'])} clubs from {source_file}"
            )

        except Exception as e:
            self.logger.error(f"Error processing club file {file_path}: {str(e)}")
            self.error_log.append(f"Club file error: {source_file}: {str(e)}")
            self.stats['errors'] += 1

    def _validate_club_data(self, club: Dict) -> bool:
        """Validate club data before insertion"""
        try:
            # Check if club has at least a name or code
            if not club.get('name') and not club.get('code'):
                self.logger.warning("Club missing both name and code")
                return False

            # If name exists, ensure it's not empty after stripping
            if club.get('name') and not str(club.get('name')).strip():
                self.logger.warning("Empty club name after stripping")
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Club validation error: {str(e)}")
            return False

    def _insert_club_data(self, club_data: Dict) -> bool:
        """Insert club data into database with error handling"""
        try:
            cursor = self.conn.cursor()

            # Ensure we have valid data
            if not club_data.get('name') and not club_data.get('code'):
                return False

            columns = list(club_data.keys())
            values = [club_data[col] for col in columns]
            placeholders = ','.join(['?' for _ in columns])

            sql = f'''
            INSERT OR REPLACE INTO clubs (
                {','.join(columns)}
            ) VALUES ({placeholders})
            '''

            cursor.execute(sql, values)
            self.conn.commit()
            return True

        except Exception as e:
            self.logger.error(f"Error inserting club data: {str(e)}")
            self.error_log.append(f"Club insertion error: {str(e)}")
            return False

    def _extract_country_from_path(self, file_path: Path) -> str:
        """Extract country from file path"""
        country_mapping = {
            'en': 'England',
            'es': 'Spain',
            'de': 'Germany',
            'it': 'Italy',
            'fr': 'France',
            'at': 'Austria',
            'be': 'Belgium',
            'nl': 'Netherlands',
            'pt': 'Portugal',
            'br': 'Brazil',
            'ru': 'Russia',
            'sco': 'Scotland',
            'ch': 'Switzerland',
            'tr': 'Turkey',
            'gr': 'Greece',
            'hu': 'Hungary',
            'cz': 'Czech Republic',
            'mx': 'Mexico'
        }

        league_code = file_path.stem.split('.')[0]
        return country_mapping.get(league_code, 'Unknown')

    def _extract_competition_from_filename(self, file_path: Path) -> str:
        """Extract competition name from filename"""
        competition_mapping = {
            'en.1': 'Premier League',
            'en.2': 'Championship',
            'en.3': 'League One',
            'en.4': 'League Two',
            'es.1': 'La Liga',
            'es.2': 'La Liga 2',
            'de.1': 'Bundesliga',
            'de.2': 'Bundesliga 2',
            'de.3': '3. Liga',
            'it.1': 'Serie A',
            'it.2': 'Serie B',
            'fr.1': 'Ligue 1',
            'fr.2': 'Ligue 2',
            'at.1': 'Austrian Bundesliga',
            'at.2': 'Austrian Erste Liga'
        }

        league_code = '.'.join(file_path.stem.split('.')[:2])
        return competition_mapping.get(league_code, league_code)

    def print_statistics(self):
        """Print detailed processing statistics"""
        print("\nProcessing Statistics:")
        print("-" * 30)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Matches processed: {self.stats['matches_processed']}")
        print(f"Clubs processed: {self.stats['clubs_processed']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Warnings encountered: {self.stats['warnings']}")

        print("\nLeagues processed:")
        for league in sorted(self.stats['leagues_processed']):
            print(f"- {league}")

        cursor = self.conn.cursor()

        print("\nDatabase Statistics:")
        print("-" * 30)

        # Matches by country
        cursor.execute("""
            SELECT 
                country, 
                COUNT(*) as total_matches,
                SUM(CASE WHEN match_status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_matches,
                ROUND(AVG(total_goals), 2) as avg_goals
            FROM matches 
            GROUP BY country 
            ORDER BY total_matches DESC
        """)
        print("\nMatches by country:")
        for row in cursor.fetchall():
            print(f"- {row[0]}: {row[1]:,} matches ({row[2]:,} completed, avg {row[3]} goals)")

        # Matches by season
        cursor.execute("""
            SELECT 
                season,
                COUNT(*) as total_matches,
                SUM(CASE WHEN match_status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_matches
            FROM matches 
            GROUP BY season 
            ORDER BY season
        """)
        print("\nMatches by season:")
        for row in cursor.fetchall():
            print(f"- {row[0]}: {row[1]:,} matches ({row[2]:,} completed)")

    def close(self):
        """Close database connection and cleanup"""
        if self.conn:
            self.conn.close()


def main():
    try:
        # Initialize processor
        processor = FootballDataProcessor(
            db_name=r'C:\Users\kyleh\Desktop\MatchMind_Master\MatchMind_db.db',
            repo_path=r'C:\Users\kyleh\Desktop\MatchMind_Master\football.json-master'
        )

        # Process all files
        processor.process_all()

        # Print statistics
        processor.print_statistics()

    except Exception as e:
        print(f"Fatal error: {str(e)}")

    finally:
        if 'processor' in locals():
            processor.close()


if __name__ == "__main__":
    main()
