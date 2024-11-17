import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np


class MatchDataExporter:
    def __init__(self, db_path: str = r"C:\Users\kyleh\Desktop\MatchMind_Master\MatchMind_db.db"):
        self.db_path = db_path
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f'data_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_match_data(self) -> pd.DataFrame:
        """Get all match data with proper null handling"""
        try:
            conn = sqlite3.connect(self.db_path)

            base_columns = """
                date,
                time,
                team1,
                team2,
                team1_score_ht,
                team2_score_ht,
                team1_score_ft,
                team2_score_ft,
                total_goals,
                goals_per_half,
                ht_result,
                ft_result,
                score_changed,
                match_status,
                over_0_5,
                over_1_5,
                over_2_5,
                over_3_5,
                over_4_5,
                over_5_5,
                under_0_5,
                under_1_5,
                under_2_5,
                under_3_5,
                under_4_5,
                under_5_5,
                goal_difference,
                clean_sheet_team1,
                clean_sheet_team2,
                btts,
                competition_type,
                country,
                competition,
                season,
                round
            """

            # Regular matches
            matches_query = f"""
            SELECT 
                'League/Cup' as source,
                {base_columns}
            FROM matches
            WHERE match_status = 'COMPLETED'
            """

            # World Cup matches
            world_cup_query = f"""
            SELECT 
                'World Cup' as source,
                {base_columns}
            FROM world_cup_matches
            WHERE match_status = 'COMPLETED'
            """

            # Euro matches
            euro_query = f"""
            SELECT 
                'Euro' as source,
                {base_columns}
            FROM euro_matches
            WHERE match_status = 'COMPLETED'
            """

            # Copa América matches
            copa_query = f"""
            SELECT 
                'Copa América' as source,
                {base_columns}
            FROM copa_matches
            WHERE match_status = 'COMPLETED'
            """

            # Read data with proper dtype specifications
            dtype_dict = {
                'source': 'category',
                'competition': 'category',
                'season': 'category',
                'round': 'category',
                'team1': str,
                'team2': str,
                'competition_type': 'category',
                'country': 'category',
                'ht_result': 'category',
                'ft_result': 'category',
                'match_status': 'category'
            }

            matches_df = pd.read_sql_query(matches_query, conn, dtype=dtype_dict)
            world_cup_df = pd.read_sql_query(world_cup_query, conn, dtype=dtype_dict)
            euro_df = pd.read_sql_query(euro_query, conn, dtype=dtype_dict)
            copa_df = pd.read_sql_query(copa_query, conn, dtype=dtype_dict)

            # Fill NaN values appropriately
            numeric_cols = [
                'team1_score_ht', 'team2_score_ht', 'team1_score_ft', 'team2_score_ft',
                'total_goals', 'goals_per_half', 'score_changed', 'over_0_5', 'over_1_5',
                'over_2_5', 'over_3_5', 'over_4_5', 'over_5_5', 'under_0_5', 'under_1_5',
                'under_2_5', 'under_3_5', 'under_4_5', 'under_5_5', 'goal_difference',
                'clean_sheet_team1', 'clean_sheet_team2', 'btts'
            ]

            categorical_cols = [
                'ht_result', 'ft_result', 'match_status'
            ]

            dfs = [matches_df, world_cup_df, euro_df, copa_df]
            for df in dfs:
                # Fill numeric columns with 0
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0).astype(int)

                # Fill categorical columns
                df['ht_result'] = df['ht_result'].fillna('PENDING')
                df['ft_result'] = df['ft_result'].fillna('PENDING')
                df['match_status'] = df['match_status'].fillna('SCHEDULED')

            # Combine DataFrames
            combined_df = pd.concat(dfs, ignore_index=True)

            # Convert date to datetime
            combined_df['date'] = pd.to_datetime(combined_df['date'])

            # Add derived features
            combined_df['year'] = combined_df['date'].dt.year
            combined_df['month'] = combined_df['date'].dt.month
            combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
            combined_df['is_weekend'] = combined_df['day_of_week'].isin([5, 6]).astype(int)

            # Add competition level
            combined_df['competition_level'] = combined_df.apply(self.determine_competition_level, axis=1)

            # Add team strength features
            combined_df = self.add_team_strength_features(combined_df)

            # Sort by date
            combined_df = combined_df.sort_values('date')

            self.logger.info(f"Exported {len(combined_df)} matches in total")
            self.logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

            return combined_df

        except Exception as e:
            self.logger.error(f"Error getting match data: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def determine_competition_level(self, row: pd.Series) -> int:
        """Determine competition level (1-5, with 1 being highest)"""
        source = row['source']
        competition = row['competition']
        competition_type = row['competition_type']

        if source in ['World Cup', 'Euro', 'Copa América']:
            return 1
        elif competition_type == 'Cup':
            if 'Champions League' in competition:
                return 1
            else:
                return 2
        else:  # League matches
            if any(league in competition for league in
                   ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']):
                return 2
            elif any(league in competition for league in
                     ['Championship', 'La Liga 2', '2. Bundesliga', 'Serie B', 'Ligue 2']):
                return 3
            elif any(league in competition for league in ['League One', '3. Liga']):
                return 4
            else:
                return 5

    def add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team strength features based on recent performance"""
        # Initialize new columns
        df['team1_form'] = 0.0
        df['team2_form'] = 0.0
        df['team1_goals_scored_avg'] = 0.0
        df['team2_goals_scored_avg'] = 0.0
        df['team1_goals_conceded_avg'] = 0.0
        df['team2_goals_conceded_avg'] = 0.0

        # Calculate rolling averages for each team
        for team in df['team1'].unique():
            # Get all matches for this team (both home and away)
            team_matches = df[(df['team1'] == team) | (df['team2'] == team)].sort_values('date')

            for idx, match in team_matches.iterrows():
                past_matches = team_matches[team_matches['date'] < match['date']].tail(5)

                if len(past_matches) > 0:
                    # Calculate form (win = 3, draw = 1, loss = 0)
                    form_points = []
                    goals_scored = []
                    goals_conceded = []

                    for _, past_match in past_matches.iterrows():
                        if past_match['team1'] == team:
                            if past_match['ft_result'] == 'WIN':
                                form_points.append(3)
                            elif past_match['ft_result'] == 'DRAW':
                                form_points.append(1)
                            else:
                                form_points.append(0)
                            goals_scored.append(past_match['team1_score_ft'])
                            goals_conceded.append(past_match['team2_score_ft'])
                        else:
                            if past_match['ft_result'] == 'LOSS':
                                form_points.append(3)
                            elif past_match['ft_result'] == 'DRAW':
                                form_points.append(1)
                            else:
                                form_points.append(0)
                            goals_scored.append(past_match['team2_score_ft'])
                            goals_conceded.append(past_match['team1_score_ft'])

                    form = np.mean(form_points) if form_points else 0
                    goals_scored_avg = np.mean(goals_scored) if goals_scored else 0
                    goals_conceded_avg = np.mean(goals_conceded) if goals_conceded else 0

                    if match['team1'] == team:
                        df.at[idx, 'team1_form'] = form
                        df.at[idx, 'team1_goals_scored_avg'] = goals_scored_avg
                        df.at[idx, 'team1_goals_conceded_avg'] = goals_conceded_avg
                    else:
                        df.at[idx, 'team2_form'] = form
                        df.at[idx, 'team2_goals_scored_avg'] = goals_scored_avg
                        df.at[idx, 'team2_goals_conceded_avg'] = goals_conceded_avg

        return df

    def export_to_csv(self, output_path: str = None):
        """Export match data to CSV with schema description"""
        try:
            # Get the data
            df = self.get_match_data()

            if df.empty:
                self.logger.error("No data to export")
                return

            # Generate default output path if none provided
            if output_path is None:
                output_path = f'match_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

            # Export to CSV
            df.to_csv(output_path, index=False)

            # Print statistics
            self.logger.info("\nExport Statistics:")
            self.logger.info("-" * 50)
            self.logger.info(f"Total matches: {len(df):,}")
            self.logger.info(f"\nMatches by source:")
            for source in df['source'].unique():
                count = len(df[df['source'] == source])
                self.logger.info(f"- {source}: {count:,}")

            self.logger.info(f"\nMatches by competition level:")
            for level in sorted(df['competition_level'].unique()):
                count = len(df[df['competition_level'] == level])
                self.logger.info(f"- Level {level}: {count:,}")

            self.logger.info(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
            self.logger.info(f"\nData exported to: {output_path}")

            # Export schema description
            schema_path = output_path.replace('.csv', '_schema.txt')
            with open(schema_path, 'w', encoding='utf-8') as f:
                f.write("Dataset Schema Description\n")
                f.write("=" * 50 + "\n\n")

                for column in df.columns:
                    f.write(f"{column}:\n")
                    f.write(f"  Type: {df[column].dtype}\n")

                    # Handle numeric columns
                    if pd.api.types.is_numeric_dtype(df[column]):
                        min_val = df[column].min()
                        max_val = df[column].max()
                        mean_val = df[column].mean()
                        if pd.notna(min_val) and pd.notna(max_val):
                            f.write(f"  Range: {min_val} to {max_val}\n")
                            f.write(f"  Mean: {mean_val:.2f}\n")

                    # Handle categorical columns
                    elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
                        unique_values = df[column].value_counts()
                        if len(unique_values) < 10:
                            f.write("  Values:\n")
                            for val, count in unique_values.items():
                                f.write(f"    - {val}: {count:,} ({(count / len(df)) * 100:.2f}%)\n")
                        else:
                            f.write(f"  Unique values: {len(unique_values):,}\n")

                    # Add null value information
                    null_count = df[column].isnull().sum()
                    if null_count > 0:
                        f.write(f"  Null values: {null_count:,} ({(null_count / len(df)) * 100:.2f}%)\n")

                    f.write("\n")

                # Add dataset summary
                f.write("\nDataset Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total records: {len(df):,}\n")
                f.write(f"Time range: {df['date'].min()} to {df['date'].max()}\n")
                f.write(f"Competitions covered: {df['competition'].nunique():,}\n")
                f.write(f"Total teams: {df['team1'].nunique() + df['team2'].nunique():,}\n")
                f.write(f"Average goals per match: {df['total_goals'].mean():.2f}\n")
                f.write(f"Most common result: {df['ft_result'].mode().iloc[0]}\n")

                # Add feature correlations with total_goals
                f.write("\nFeature Correlations with Total Goals:\n")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                correlations = df[numeric_cols].corr()['total_goals'].sort_values(ascending=False)

                for col, corr in correlations.items():
                    if col != 'total_goals':
                        f.write(f"  {col}: {corr:.3f}\n")

                # Add some key statistics about different competition levels
                f.write("\nGoals by Competition Level:\n")
                level_stats = df.groupby('competition_level')['total_goals'].agg(['mean', 'std', 'count'])
                for level, stats in level_stats.iterrows():
                    f.write(f"  Level {level}:\n")
                    f.write(f"    Average goals: {stats['mean']:.2f}\n")
                    f.write(f"    Standard deviation: {stats['std']:.2f}\n")
                    f.write(f"    Number of matches: {stats['count']:,}\n")

                # Add home vs away team statistics
                f.write("\nHome vs Away Team Statistics:\n")
                f.write(f"  Home team wins: {len(df[df['ft_result'] == 'WIN']):,} ")
                f.write(f"({len(df[df['ft_result'] == 'WIN']) / len(df) * 100:.2f}%)\n")
                f.write(f"  Away team wins: {len(df[df['ft_result'] == 'LOSS']):,} ")
                f.write(f"({len(df[df['ft_result'] == 'LOSS']) / len(df) * 100:.2f}%)\n")
                f.write(f"  Draws: {len(df[df['ft_result'] == 'DRAW']):,} ")
                f.write(f"({len(df[df['ft_result'] == 'DRAW']) / len(df) * 100:.2f}%)\n")

                # Add over/under statistics
                f.write("\nOver/Under Statistics:\n")
                for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    over_count = len(df[df[f'over_{str(threshold).replace(".", "_")}'] == 1])
                    under_count = len(df[df[f'under_{str(threshold).replace(".", "_")}'] == 1])
                    f.write(f"  Over {threshold}: {over_count:,} ({over_count / len(df) * 100:.2f}%)\n")
                    f.write(f"  Under {threshold}: {under_count:,} ({under_count / len(df) * 100:.2f}%)\n")

                # Add BTTS (Both Teams To Score) statistics
                btts_count = len(df[df['btts'] == 1])
                f.write(f"\nBTTS Statistics:\n")
                f.write(f"  Both teams scored: {btts_count:,} ({btts_count / len(df) * 100:.2f}%)\n")
                f.write(
                    f"  Clean sheets: {len(df[(df['clean_sheet_team1'] == 1) | (df['clean_sheet_team2'] == 1)]):,} ")
                f.write(
                    f"({len(df[(df['clean_sheet_team1'] == 1) | (df['clean_sheet_team2'] == 1)]) / len(df) * 100:.2f}%)\n")

            self.logger.info(f"Schema description exported to: {schema_path}")

            except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            raise

def main():
        try:
            exporter = MatchDataExporter()
            exporter.export_to_csv('full_match_data.csv')
        except Exception as e:
            print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
        main()