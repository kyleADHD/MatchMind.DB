import sqlite3
import json
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import time
from dataclasses import dataclass


@dataclass
class LeagueInfo:
    api_id: str
    name: str
    country: str


class APIConfig:
    def __init__(self):
        self.current_season = "2023-24"
        # Mapping of league names to their API identifiers and metadata
        self.leagues = {
            # Major European Leagues
            "en.1": LeagueInfo("en.1", "Premier League", "England"),
            "en.2": LeagueInfo("en.2", "Championship", "England"),
            "en.3": LeagueInfo("en.3", "League One", "England"),
            "en.4": LeagueInfo("en.4", "League Two", "England"),
            "es.1": LeagueInfo("es.1", "Primera División", "Spain"),
            "es.2": LeagueInfo("es.2", "Segunda División", "Spain"),
            "de.1": LeagueInfo("de.1", "Bundesliga", "Germany"),
            "de.2": LeagueInfo("de.2", "2. Bundesliga", "Germany"),
            "de.3": LeagueInfo("de.3", "3. Liga", "Germany"),
            "it.1": LeagueInfo("it.1", "Serie A", "Italy"),
            "it.2": LeagueInfo("it.2", "Serie B", "Italy"),
            "fr.1": LeagueInfo("fr.1", "Ligue 1", "France"),
            "fr.2": LeagueInfo("fr.2", "Ligue 2", "France"),

            # Other European Leagues
            "nl.1": LeagueInfo("nl.1", "Eredivisie", "Netherlands"),
            "pt.1": LeagueInfo("pt.1", "Primeira Liga", "Portugal"),
            "be.1": LeagueInfo("be.1", "First Division A", "Belgium"),
            "tr.1": LeagueInfo("tr.1", "Süper Lig", "Turkey"),
            "at.1": LeagueInfo("at.1", "Bundesliga", "Austria"),
            "at.2": LeagueInfo("at.2", "2. Liga", "Austria"),
            "ch.1": LeagueInfo("ch.1", "Super League", "Switzerland"),
            "ch.2": LeagueInfo("ch.2", "Challenge League", "Switzerland"),

            # Other Leagues
            "br.1": LeagueInfo("br.1", "Campeonato Brasileiro Série A", "Brazil"),
            "jp.1": LeagueInfo("jp.1", "J. League", "Japan"),
            "mx.1": LeagueInfo("mx.1", "Primera División", "Mexico"),
            "au.1": LeagueInfo("au.1", "A-League", "Australia"),

            # Cups and International Competitions
            "cl": LeagueInfo("cl", "UEFA Champions League", "Europe"),
            "de.cup": LeagueInfo("de.cup", "DFB Pokal", "Germany"),
            "at.cup": LeagueInfo("at.cup", "ÖFB Cup", "Austria")
        }

        self.base_url = "https://raw.githubusercontent.com/openfootball/football.json/master"


class APIDBSync:
    def __init__(self, db_path: str = 'MatchMind_db.db'):
        self.db_path = db_path
        self.api_config = APIConfig()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f'api_sync_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_api_data(self, league_id: str) -> Optional[Dict]:
        """Fetch data from the API for the current season and specified league"""
        try:
            season = self.api_config.current_season
            url = f"{self.api_config.base_url}/{season}/{league_id}.json"
            self.logger.info(f"Fetching data from: {url}")

            try:
                response = requests.get(url)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed: {str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching API data: {str(e)}")
            return None

    def get_db_competition_name(self, api_league_id: str) -> str:
        """Get standardized competition name from league ID"""
        league_info = self.api_config.leagues.get(api_league_id)
        return league_info.name if league_info else "Unknown"

    def get_existing_matches(self, competition: str) -> Dict[str, Dict]:
        """Get existing matches from database for the current season"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
            SELECT id, date, team1, team2, team1_score_ft, team2_score_ft,
                   team1_score_ht, team2_score_ht, last_updated
            FROM matches
            WHERE season = ? AND competition = ?
            """

            cursor.execute(query, (self.api_config.current_season, competition))
            matches = {}

            for row in cursor.fetchall():
                key = f"{row[1]}_{row[2]}_{row[3]}"  # date_team1_team2
                matches[key] = {
                    'id': row[0],
                    'date': row[1],
                    'team1': row[2],
                    'team2': row[3],
                    'score_ft': (row[4], row[5]),
                    'score_ht': (row[6], row[7]),
                    'last_updated': row[8]
                }

            return matches

        except Exception as e:
            self.logger.error(f"Error getting existing matches: {str(e)}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()

    def sync_league(self, league_id: str) -> Tuple[int, int, int]:
        """Sync matches for a specific league in the current season"""
        updates = 0
        inserts = 0
        errors = 0

        try:
            # Get league information
            league_info = self.api_config.leagues[league_id]
            if not league_info:
                self.logger.error(f"Unknown league ID: {league_id}")
                return 0, 0, 1

            # Fetch API data
            api_data = self.fetch_api_data(league_id)
            if not api_data:
                return 0, 0, 1

            # Get existing matches
            existing_matches = self.get_existing_matches(league_info.name)

            # Process matches
            matches = []
            if 'matches' in api_data:
                matches = api_data['matches']
            elif 'rounds' in api_data:
                for round_data in api_data['rounds']:
                    if 'matches' in round_data:
                        for match in round_data['matches']:
                            match['round'] = round_data.get('name', '')
                            matches.append(match)

            # Update or insert matches
            for match in matches:
                try:
                    processed_match = self.process_api_match(
                        match,
                        self.api_config.current_season,
                        league_info.name,
                        league_info.country
                    )

                    if not processed_match:
                        errors += 1
                        continue

                    match_key = f"{match['date']}_{match['team1']}_{match['team2']}"
                    existing_match = existing_matches.get(match_key)

                    if existing_match:
                        # Update if scores are different
                        if (processed_match['team1_score_ft'], processed_match['team2_score_ft']) != existing_match[
                            'score_ft']:
                            if self.update_database(processed_match, existing_match['id']):
                                updates += 1
                            else:
                                errors += 1
                    else:
                        # Insert new match
                        if self.update_database(processed_match):
                            inserts += 1
                        else:
                            errors += 1

                except Exception as e:
                    self.logger.error(f"Error processing match: {str(e)}")
                    errors += 1

            return updates, inserts, errors

        except Exception as e:
            self.logger.error(f"Error syncing league {league_id}: {str(e)}")
            return updates, inserts, errors + 1

    def sync_all(self) -> Dict:
        """Sync all configured leagues for the current season"""
        stats = {
            'total_updates': 0,
            'total_inserts': 0,
            'total_errors': 0,
            'leagues_processed': [],
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None
        }

        try:
            for league_id, league_info in self.api_config.leagues.items():
                self.logger.info(f"Syncing {league_info.name}")
                updates, inserts, errors = self.sync_league(league_id)

                stats['total_updates'] += updates
                stats['total_inserts'] += inserts
                stats['total_errors'] += errors

                if updates + inserts > 0:
                    stats['leagues_processed'].append(league_info.name)

                # Add delay between API calls to avoid rate limiting
                time.sleep(2)

        except Exception as e:
            self.logger.error(f"Error in sync_all: {str(e)}")
        finally:
            stats['end_time'] = datetime.now()
            stats['duration'] = stats['end_time'] - stats['start_time']

        return stats


def main():
    try:
        sync = APIDBSync()

        # Run full sync
        stats = sync.sync_all()

        # Print results
        print("\nSync Complete!")
        print("-" * 50)
        print(f"Total Updates: {stats['total_updates']}")
        print(f"Total Inserts: {stats['total_inserts']}")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Duration: {stats['duration']}")
        print("\nLeagues processed:")
        for league in sorted(stats['leagues_processed']):
            print(f"- {league}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()