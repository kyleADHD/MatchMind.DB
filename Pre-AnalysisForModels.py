# import sqlite3
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from dataclasses import dataclass
# from datetime import datetime
# import scipy.stats as stats
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# @dataclass
# class MarketAnalysis:
#     name: str
#     query: str
#     features: List[str]
#     description: str
#
#
# class BettingMarketAnalyzer:
#     def __init__(self, db_path: str = 'MatchMind_db.db'):
#         self.db_path = db_path
#         self.markets = self._initialize_markets()
#
#     def _initialize_markets(self) -> Dict[str, MarketAnalysis]:
#         return {
#             '1X2': MarketAnalysis(
#                 name='Match Result (1X2)',
#                 query="""
#                     SELECT
#                         competition, season, date,
#                         team1, team2, ft_result,
#                         team1_score_ht, team2_score_ht,
#                         team1_score_ft, team2_score_ft,
#                         total_goals, goals_per_half,
#                         competition_type, country
#                     FROM matches
#                     WHERE match_status = 'COMPLETED'
#                 """,
#                 features=['ft_result', 'total_goals', 'goals_per_half'],
#                 description='Home/Draw/Away market analysis'
#             ),
#             'BTTS': MarketAnalysis(
#                 name='Both Teams to Score',
#                 query="""
#                     SELECT
#                         competition, season, date,
#                         team1, team2, btts,
#                         team1_score_ht, team2_score_ht,
#                         clean_sheet_team1, clean_sheet_team2,
#                         competition_type, country
#                     FROM matches
#                     WHERE match_status = 'COMPLETED'
#                 """,
#                 features=['btts', 'clean_sheet_team1', 'clean_sheet_team2'],
#                 description='Both teams to score analysis'
#             ),
#             'OVER_UNDER': MarketAnalysis(
#                 name='Over/Under Goals',
#                 query="""
#                     SELECT
#                         competition, season, date,
#                         team1, team2, total_goals,
#                         over_0_5, over_1_5, over_2_5,
#                         over_3_5, over_4_5, over_5_5,
#                         under_0_5, under_1_5, under_2_5,
#                         under_3_5, under_4_5, under_5_5,
#                         competition_type, country
#                     FROM matches
#                     WHERE match_status = 'COMPLETED'
#                 """,
#                 features=['total_goals', 'over_2_5', 'under_2_5'],
#                 description='Over/Under goals market analysis'
#             )
#         }
#
#     def load_market_data(self, market_key: str) -> pd.DataFrame:
#         try:
#             conn = sqlite3.connect(self.db_path)
#             market = self.markets[market_key]
#             df = pd.read_sql_query(market.query, conn)
#             conn.close()
#             return df
#         except Exception as e:
#             print(f"Error loading {market_key} data: {str(e)}")
#             return pd.DataFrame()
#
#     def create_key_visualizations(self):
#         df_1x2 = self.load_market_data('1X2')
#         df_btts = self.load_market_data('BTTS')
#         df_ou = self.load_market_data('OVER_UNDER')
#
#         fig = make_subplots(
#             rows=2, cols=2,
#             subplot_titles=(
#                 'Match Result Distribution',
#                 'BTTS Rate by Competition',
#                 'Goals Distribution',
#                 'Top Feature Correlations'
#             )
#         )
#
#         result_dist = df_1x2['ft_result'].value_counts()
#         fig.add_trace(
#             go.Bar(x=result_dist.index, y=result_dist.values, name='Match Results'),
#             row=1, col=1
#         )
#
#         btts_rate = df_btts.groupby('competition')['btts'].mean().sort_values(ascending=False).head(10)
#         fig.add_trace(
#             go.Bar(x=btts_rate.index, y=btts_rate.values, name='BTTS Rate'),
#             row=1, col=2
#         )
#
#         fig.add_trace(
#             go.Histogram(x=df_ou['total_goals'], name='Goals Distribution', nbinsx=10),
#             row=2, col=1
#         )
#
#         numeric_cols = df_ou.select_dtypes(include=[np.number]).columns
#         corr_matrix = df_ou[numeric_cols].corr()
#
#         correlations = []
#         for i in range(len(corr_matrix.columns)):
#             for j in range(i + 1, len(corr_matrix.columns)):
#                 correlations.append({
#                     'pair': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
#                     'correlation': abs(corr_matrix.iloc[i, j])
#                 })
#
#         top_corr = pd.DataFrame(correlations).nlargest(5, 'correlation')
#
#         fig.add_trace(
#             go.Bar(
#                 x=top_corr['correlation'],
#                 y=top_corr['pair'],
#                 orientation='h',
#                 name='Top Correlations'
#             ),
#             row=2, col=2
#         )
#
#         fig.update_layout(
#             height=800,
#             width=1200,
#             showlegend=False,
#             title_text="Key Betting Market Metrics",
#             template="plotly_white"
#         )
#
#         return fig
#
#     def print_key_insights(self):
#         df_1x2 = self.load_market_data('1X2')
#         df_btts = self.load_market_data('BTTS')
#         df_ou = self.load_market_data('OVER_UNDER')
#
#         print("=== Key Betting Market Insights ===")
#
#         result_dist = df_1x2['ft_result'].value_counts(normalize=True)
#         print("\nMatch Result Distribution:")
#         for result, pct in result_dist.items():
#             print(f"{result}: {pct:.1%}")
#
#         btts_rate = df_btts['btts'].mean()
#         print(f"\nOverall BTTS Rate: {btts_rate:.1%}")
#
#         avg_goals = df_ou['total_goals'].mean()
#         over_25_rate = df_ou['over_2_5'].mean()
#         print(f"\nGoals per Match: {avg_goals:.2f}")
#         print(f"Over 2.5 Goals Rate: {over_25_rate:.1%}")
#
#
# class ModelPreAnalyzer:
#     def __init__(self, db_path: str = 'MatchMind_db.db'):
#         self.db_path = db_path
#         self.markets = {
#             '1X2': {'target': 'ft_result', 'type': 'categorical'},
#             'BTTS': {'target': 'btts', 'type': 'binary'},
#             'OVER_UNDER': {'target': 'over_2_5', 'type': 'binary'}
#         }
#         self.betting_analyzer = BettingMarketAnalyzer(db_path)
#
#     def print_detailed_recommendations(self, recommendations):
#         """Print detailed recommendations with explanations"""
#         print("\n=== Detailed Modeling Recommendations ===")
#
#         for category, recs in recommendations.items():
#             if recs:
#                 print(f"\n{category.replace('_', ' ').title()}:")
#                 for i, rec in enumerate(recs, 1):
#                     print(f"{i}. {rec}")
#
#         if not any(recommendations.values()):
#             print("\nNo specific recommendations needed - data looks good for modeling!")
#
#     def analyze_market(self, market_name: str):
#         conn = sqlite3.connect(self.db_path)
#         query = f"""
#             SELECT
#                 m.*,
#                 strftime('%Y', date) as year,
#                 strftime('%m', date) as month,
#                 strftime('%w', date) as day_of_week,
#                 julianday(date) - julianday(date, 'start of month') + 1 as day_of_month
#             FROM matches m
#             WHERE match_status = 'COMPLETED'
#         """
#         df = pd.read_sql_query(query, conn)
#         conn.close()
#
#         market_info = self.markets[market_name]
#         target = market_info['target']
#
#         quality_analysis = self._analyze_data_quality(df)
#         target_analysis = self._analyze_target(df, target, market_info['type'])
#         feature_analysis = self._analyze_features(df, target, market_info['type'])
#         time_analysis = self._analyze_time_patterns(df, target)
#         market_insights = self._analyze_market_specific(df, market_name)
#
#         # Combine betting market analysis
#         betting_analysis = self.betting_analyzer.load_market_data(market_name)
#
#         dashboard = self._create_analysis_dashboard(
#             df, target, market_info['type'],
#             quality_analysis, target_analysis,
#             feature_analysis, time_analysis,
#             market_insights
#         )
#
#         return {
#             'data_quality': quality_analysis,
#             'target_analysis': target_analysis,
#             'feature_analysis': feature_analysis,
#             'time_analysis': time_analysis,
#             'market_insights': market_insights,
#             'betting_analysis': betting_analysis,
#             'dashboard': dashboard
#         }
#
#     def _analyze_data_quality(self, df):
#         total_rows = len(df)
#
#         return {
#             'total_samples': total_rows,
#             'missing_values': df.isnull().sum().to_dict(),
#             'missing_pct': (df.isnull().sum() / total_rows * 100).to_dict(),
#             'duplicates': df.duplicated().sum(),
#             'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
#             'categorical_columns': list(df.select_dtypes(include=['object']).columns)
#         }
#
#     def _analyze_target(self, df, target, target_type):
#         if target_type == 'categorical':
#             distribution = df[target].value_counts(normalize=True)
#             class_balance = distribution.to_dict()
#             majority_class = distribution.index[0]
#             imbalance_ratio = distribution.max() / distribution.min()
#         else:
#             distribution = df[target].mean()
#             class_balance = {'positive': distribution, 'negative': 1 - distribution}
#             majority_class = 'negative' if distribution < 0.5 else 'positive'
#             imbalance_ratio = max(distribution, 1 - distribution) / min(distribution, 1 - distribution)
#
#         return {
#             'distribution': distribution,
#             'class_balance': class_balance,
#             'majority_class': majority_class,
#             'imbalance_ratio': imbalance_ratio
#         }
#
#     def _analyze_features(self, df, target, target_type):
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         numeric_cols = [col for col in numeric_cols if col != target]
#
#         correlations = {}
#         if target_type == 'categorical':
#             target_dummies = pd.get_dummies(df[target], prefix=target)
#             for col in numeric_cols:
#                 max_corr = max(abs(df[col].corr(target_dummies[dummy]))
#                                for dummy in target_dummies.columns)
#                 correlations[col] = max_corr
#         else:
#             for col in numeric_cols:
#                 correlations[col] = abs(df[col].corr(df[target]))
#
#         distributions = {}
#         for col in numeric_cols:
#             distributions[col] = {
#                 'mean': df[col].mean(),
#                 'std': df[col].std(),
#                 'skew': df[col].skew(),
#                 'kurtosis': df[col].kurtosis(),
#                 'normality_test': stats.normaltest(df[col].dropna())[1]
#             }
#
#         correlation_matrix = df[numeric_cols].corr()
#         high_correlations = []
#         for i in range(len(numeric_cols)):
#             for j in range(i + 1, len(numeric_cols)):
#                 if abs(correlation_matrix.iloc[i, j]) > 0.8:
#                     high_correlations.append({
#                         'feature1': numeric_cols[i],
#                         'feature2': numeric_cols[j],
#                         'correlation': correlation_matrix.iloc[i, j]
#                     })
#
#         return {
#             'feature_importance': correlations,
#             'distributions': distributions,
#             'high_correlations': high_correlations
#         }
#
#     def _analyze_time_patterns(self, df, target):
#         """Analyze temporal patterns in the target variable"""
#         df['date'] = pd.to_datetime(df['date'])
#
#         # Handle categorical targets differently from numerical ones
#         if df[target].dtype == 'object':  # Categorical target
#             # For each time period, calculate distribution of categories
#             monthly_pattern = df.groupby('month')[target].value_counts(normalize=True).unstack()
#             yearly_pattern = df.groupby('year')[target].value_counts(normalize=True).unstack()
#             dow_pattern = df.groupby('day_of_week')[target].value_counts(normalize=True).unstack()
#
#             return {
#                 'monthly_pattern': monthly_pattern,
#                 'yearly_pattern': yearly_pattern,
#                 'day_of_week_pattern': dow_pattern,
#                 'is_categorical': True
#             }
#         else:  # Numerical target
#             monthly_pattern = df.groupby('month')[target].mean()
#             yearly_pattern = df.groupby('year')[target].mean()
#             dow_pattern = df.groupby('day_of_week')[target].mean()
#
#             return {
#                 'monthly_pattern': monthly_pattern,
#                 'yearly_pattern': yearly_pattern,
#                 'day_of_week_pattern': dow_pattern,
#                 'is_categorical': False
#             }
#
#     def _analyze_market_specific(self, df, market_name):
#         """Market-specific analysis based on domain knowledge"""
#         insights = {}
#
#         if market_name == '1X2':
#             insights['home_advantage'] = (df['ft_result'] == 'WIN').mean()
#             insights['draw_rate'] = (df['ft_result'] == 'DRAW').mean()
#             insights['home_scoring_rate'] = (df['team1_score_ft'] > 0).mean()
#
#         elif market_name == 'BTTS':
#             insights['first_half_btts'] = ((df['team1_score_ht'] > 0) &
#                                            (df['team2_score_ht'] > 0)).mean()
#             insights['second_half_btts'] = ((df['team1_score_ft'] - df['team1_score_ht'] > 0) &
#                                             (df['team2_score_ft'] - df['team2_score_ht'] > 0)).mean()
#
#         elif market_name == 'OVER_UNDER':
#             insights['goals_per_half'] = {
#                 'first_half': (df['team1_score_ht'] + df['team2_score_ht']).mean(),
#                 'second_half': ((df['team1_score_ft'] - df['team1_score_ht']) +
#                                 (df['team2_score_ft'] - df['team2_score_ht'])).mean()
#             }
#             insights['over_2_5_by_ht_goals'] = df.groupby(
#                 df['team1_score_ht'] + df['team2_score_ht']
#             )['over_2_5'].mean().to_dict()
#
#         return insights
#
#     def get_modeling_recommendations(self, analysis_results):
#         """Generate modeling recommendations based on analysis"""
#         recommendations = {
#             'data_preparation': [],
#             'feature_engineering': [],
#             'model_selection': [],
#             'training_approach': []
#         }
#
#         # Data preparation recommendations
#         quality = analysis_results['data_quality']
#         if any(pct > 5 for pct in quality['missing_pct'].values()):
#             recommendations['data_preparation'].append(
#                 "Handle missing values - consider imputation"
#             )
#
#         # Feature recommendations
#         feature_analysis = analysis_results['feature_analysis']
#         if feature_analysis['high_correlations']:
#             recommendations['feature_engineering'].append(
#                 "Handle multicollinearity - consider feature selection or PCA"
#             )
#
#         for feature, dist in feature_analysis['distributions'].items():
#             if dist['normality_test'] < 0.05:
#                 recommendations['feature_engineering'].append(
#                     f"Consider transforming {feature} - non-normal distribution"
#                 )
#
#         # Model selection recommendations
#         target_analysis = analysis_results['target_analysis']
#         if target_analysis['imbalance_ratio'] > 3:
#             recommendations['model_selection'].append(
#                 "Handle class imbalance - consider SMOTE or class weights"
#             )
#
#         # Training approach recommendations
#         time_analysis = analysis_results['time_analysis']
#         if time_analysis['is_categorical']:
#             # For categorical targets, check variation in distributions
#             yearly_pattern = time_analysis['yearly_pattern']
#             if not yearly_pattern.empty:
#                 std_by_category = yearly_pattern.std()
#                 if (std_by_category > 0.1).any():
#                     recommendations['training_approach'].append(
#                         "Consider temporal validation - significant yearly variations in class distribution"
#                     )
#         else:
#             # For numerical targets
#             yearly_pattern = time_analysis['yearly_pattern']
#             if not yearly_pattern.empty:
#                 if yearly_pattern.std() > 0.1:
#                     recommendations['training_approach'].append(
#                         "Consider temporal validation - significant yearly variations in target"
#                     )
#
#         # Market-specific recommendations
#         if 'market_insights' in analysis_results:
#             market_insights = analysis_results['market_insights']
#
#             # For 1X2 market
#             if 'home_advantage' in market_insights:
#                 home_adv = market_insights['home_advantage']
#                 if home_adv > 0.45:
#                     recommendations['feature_engineering'].append(
#                         "Consider creating home advantage features"
#                     )
#
#             # For BTTS market
#             if 'first_half_btts' in market_insights:
#                 if abs(market_insights['first_half_btts'] - market_insights['second_half_btts']) > 0.1:
#                     recommendations['feature_engineering'].append(
#                         "Consider separate first/second half BTTS features"
#                     )
#
#             # For Over/Under market
#             if isinstance(market_insights.get('goals_per_half'), dict):
#                 first_half = market_insights['goals_per_half']['first_half']
#                 second_half = market_insights['goals_per_half']['second_half']
#                 if abs(first_half - second_half) > 0.3:
#                     recommendations['feature_engineering'].append(
#                         "Consider separate half-specific goal features"
#                     )
#
#         # Add feature interaction recommendations
#         if feature_analysis['high_correlations']:
#             high_corr_pairs = [
#                 f"{corr['feature1']} × {corr['feature2']}"
#                 for corr in feature_analysis['high_correlations'][:3]
#             ]
#             if high_corr_pairs:
#                 recommendations['feature_engineering'].append(
#                     f"Consider interaction features between: {', '.join(high_corr_pairs)}"
#                 )
#
#         return recommendations
#
#     def _create_analysis_dashboard(self, df, target, target_type,
#                                    quality_analysis, target_analysis,
#                                    feature_analysis, time_analysis,
#                                    market_insights):
#         """Create analysis dashboard with proper subplot specifications"""
#         # Create subplots with appropriate specifications
#         fig = make_subplots(
#             rows=3, cols=2,
#             subplot_titles=(
#                 'Target Distribution',
#                 'Feature Importance',
#                 'Time Series Patterns',
#                 'Feature Correlations',
#                 'Market-Specific Insights',
#                 'Data Quality Summary'
#             ),
#             specs=[
#                 [{"type": "xy"}, {"type": "xy"}],  # First row
#                 [{"type": "xy"}, {"type": "xy"}],  # Second row
#                 [{"type": "xy"}, {"type": "xy"}]  # Third row
#             ]
#         )
#
#         # 1. Target Distribution (Row 1, Col 1)
#         if target_type == 'categorical':
#             # Bar chart for categorical targets
#             fig.add_trace(
#                 go.Bar(
#                     x=target_analysis['distribution'].index,
#                     y=target_analysis['distribution'].values,
#                     name='Distribution'
#                 ),
#                 row=1, col=1
#             )
#         else:
#             # Histogram for numerical targets
#             target_values = df[target].dropna()
#             fig.add_trace(
#                 go.Histogram(
#                     x=target_values,
#                     name='Distribution'
#                 ),
#                 row=1, col=1
#             )
#
#         # 2. Feature Importance (Row 1, Col 2)
#         importance_df = pd.DataFrame({
#             'feature': feature_analysis['feature_importance'].keys(),
#             'importance': feature_analysis['feature_importance'].values()
#         }).sort_values('importance', ascending=True).tail(10)
#
#         fig.add_trace(
#             go.Bar(
#                 x=importance_df['importance'],
#                 y=importance_df['feature'],
#                 orientation='h',
#                 name='Feature Importance'
#             ),
#             row=1, col=2
#         )
#
#         # 3. Time Series Patterns (Row 2, Col 1)
#         if time_analysis['is_categorical']:
#             # For categorical targets, plot stacked bars
#             for category in time_analysis['monthly_pattern'].columns:
#                 fig.add_trace(
#                     go.Bar(
#                         name=category,
#                         x=time_analysis['monthly_pattern'].index,
#                         y=time_analysis['monthly_pattern'][category],
#                     ),
#                     row=2, col=1
#                 )
#             fig.update_layout(barmode='stack')
#         else:
#             # For numerical targets, plot line
#             fig.add_trace(
#                 go.Scatter(
#                     x=time_analysis['monthly_pattern'].index,
#                     y=time_analysis['monthly_pattern'].values,
#                     mode='lines+markers',
#                     name='Monthly Pattern'
#                 ),
#                 row=2, col=1
#             )
#
#         # 4. Feature Correlations (Row 2, Col 2)
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         correlation_matrix = df[numeric_cols].corr()
#
#         fig.add_trace(
#             go.Heatmap(
#                 z=correlation_matrix.values,
#                 x=correlation_matrix.columns,
#                 y=correlation_matrix.columns,
#                 colorscale='RdBu',
#                 showscale=True
#             ),
#             row=2, col=2
#         )
#
#         # 5. Market-Specific Insights (Row 3, Col 1)
#         if isinstance(market_insights, dict):
#             market_data = []
#             market_values = []
#             for key, value in market_insights.items():
#                 if isinstance(value, (int, float)):
#                     market_data.append(key)
#                     market_values.append(value)
#
#             fig.add_trace(
#                 go.Bar(
#                     x=market_values,
#                     y=market_data,
#                     orientation='h',
#                     name='Market Insights'
#                 ),
#                 row=3, col=1
#             )
#
#         # 6. Data Quality Summary (Row 3, Col 2)
#         quality_metrics = pd.Series(quality_analysis['missing_pct']).sort_values(ascending=True)
#         fig.add_trace(
#             go.Bar(
#                 x=quality_metrics.values,
#                 y=quality_metrics.index,
#                 orientation='h',
#                 name='Missing Data (%)'
#             ),
#             row=3, col=2
#         )
#
#         # Update layout
#         fig.update_layout(
#             height=1200,
#             width=1200,
#             showlegend=True,
#             title_text=f"Pre-modeling Analysis Dashboard",
#             template="plotly_white"
#         )
#
#         # Update axes labels
#         fig.update_xaxes(title_text="Value", row=1, col=1)
#         fig.update_yaxes(title_text="Count", row=1, col=1)
#
#         fig.update_xaxes(title_text="Importance Score", row=1, col=2)
#         fig.update_yaxes(title_text="Feature", row=1, col=2)
#
#         fig.update_xaxes(title_text="Month", row=2, col=1)
#         fig.update_yaxes(title_text="Value", row=2, col=1)
#
#         fig.update_xaxes(title_text="Features", row=2, col=2)
#         fig.update_yaxes(title_text="Features", row=2, col=2)
#
#         fig.update_xaxes(title_text="Value", row=3, col=1)
#         fig.update_yaxes(title_text="Metric", row=3, col=1)
#
#         fig.update_xaxes(title_text="Missing Data (%)", row=3, col=2)
#         fig.update_yaxes(title_text="Feature", row=3, col=2)
#
#         return fig
#
#
# def main():
#     db_path = 'MatchMind_db.db'
#     analyzer = ModelPreAnalyzer(db_path)
#
#     for market in ['1X2', 'BTTS', 'OVER_UNDER']:
#         print(f"\n=== Analyzing {market} Market ===")
#
#         # Run analysis
#         results = analyzer.analyze_market(market)
#
#         # Display dashboard
#         results['dashboard'].show()
#
#         # Get and print recommendations
#         recommendations = analyzer.get_modeling_recommendations(results)
#         print("\nModeling Recommendations:")
#         for category, recs in recommendations.items():
#             if recs:
#                 print(f"\n{category.replace('_', ' ').title()}:")
#                 for rec in recs:
#                     print(f"- {rec}")
#
#
# if __name__ == "__main__":
#     main()
import sqlite3
from typing import Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go

class DoubleChanceAnalyzer:
    def __init__(self, db_path: str = 'MatchMind_db.db'):
        self.db_path = db_path

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for Double Chance analysis"""
        query = """
            WITH match_data AS (
                SELECT 
                    *,
                    CASE
                        WHEN team1_score_ht > team2_score_ht THEN 'HOME_LEAD'
                        WHEN team1_score_ht < team2_score_ht THEN 'AWAY_LEAD'
                        ELSE 'DRAW'
                    END as ht_result,
                    CASE
                        WHEN ft_result = 'WIN' OR ft_result = 'DRAW' THEN 1
                        ELSE 0
                    END as dc_1X,
                    CASE
                        WHEN ft_result = 'DRAW' OR ft_result = 'LOSS' THEN 1
                        ELSE 0
                    END as dc_X2,
                    CASE
                        WHEN ft_result = 'WIN' OR ft_result = 'LOSS' THEN 1
                        ELSE 0
                    END as dc_12,
                    strftime('%Y', date) as year,
                    strftime('%m', date) as month,
                    strftime('%w', date) as day_of_week
                FROM matches
                WHERE match_status = 'COMPLETED'
            )
            SELECT * FROM match_data
        """

        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("Warning: No data returned from query")
                return pd.DataFrame()

            # Verify required columns exist
            required_columns = ['dc_1X', 'dc_X2', 'dc_12', 'ht_result',
                                'competition', 'season', 'total_goals']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                return pd.DataFrame()

            return df

        except sqlite3.Error as e:
            print(f"SQLite error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()
        finally:
            if 'conn' in locals():
                conn.close()

    def calculate_dc_probabilities(self, df: pd.DataFrame) -> Dict:
        """Calculate Double Chance probabilities across different conditions"""
        if df.empty:
            print("No data available for probability calculations")
            return {}

        try:
            probabilities = {
                'overall': {
                    '1X': df['dc_1X'].mean(),
                    'X2': df['dc_X2'].mean(),
                    '12': df['dc_12'].mean()
                },
                'by_competition': {},
                'by_ht_result': {},
                'by_total_goals': {},
                'by_season': {}
            }

            # By competition
            for comp in df['competition'].unique():
                comp_df = df[df['competition'] == comp]
                if len(comp_df) > 0:
                    probabilities['by_competition'][comp] = {
                        '1X': comp_df['dc_1X'].mean(),
                        'X2': comp_df['dc_X2'].mean(),
                        '12': comp_df['dc_12'].mean()
                    }

            # By HT result
            for ht_res in df['ht_result'].unique():
                ht_df = df[df['ht_result'] == ht_res]
                if len(ht_df) > 0:
                    probabilities['by_ht_result'][ht_res] = {
                        '1X': ht_df['dc_1X'].mean(),
                        'X2': ht_df['dc_X2'].mean(),
                        '12': ht_df['dc_12'].mean()
                    }

            # By total goals range
            df['goals_range'] = pd.cut(df['total_goals'],
                                       bins=[0, 1, 2, 3, 4, float('inf')],
                                       labels=['0-1', '1-2', '2-3', '3-4', '4+'])

            for goals_range in df['goals_range'].unique():
                if pd.isna(goals_range):
                    continue
                goals_df = df[df['goals_range'] == goals_range]
                if len(goals_df) > 0:
                    probabilities['by_total_goals'][str(goals_range)] = {
                        '1X': goals_df['dc_1X'].mean(),
                        'X2': goals_df['dc_X2'].mean(),
                        '12': goals_df['dc_12'].mean()
                    }

            # By season
            for season in df['season'].unique():
                season_df = df[df['season'] == season]
                if len(season_df) > 0:
                    probabilities['by_season'][season] = {
                        '1X': season_df['dc_1X'].mean(),
                        'X2': season_df['dc_X2'].mean(),
                        '12': season_df['dc_12'].mean()
                    }

            return probabilities

        except Exception as e:
            print(f"Error calculating probabilities: {str(e)}")
            return {}

    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze specific patterns in Double Chance outcomes"""
        if df.empty:
            print("No data available for pattern analysis")
            return {}

        try:
            patterns = {
                'ht_ft_patterns': {},
                'goal_patterns': {},
                'competition_patterns': {},
                'seasonal_patterns': {},
                'streak_patterns': {},
                'score_patterns': {}
            }

            # HT-FT patterns
            ht_ft_matrix = pd.crosstab(df['ht_result'],
                                       df['ft_result'],
                                       normalize='index')
            patterns['ht_ft_patterns'] = {
                'matrix': ht_ft_matrix.to_dict(),
                'insights': {
                    'home_lead_conversion': ht_ft_matrix.loc[
                        'HOME_LEAD', 'WIN'] if 'HOME_LEAD' in ht_ft_matrix.index else 0,
                    'away_lead_conversion': ht_ft_matrix.loc[
                        'AWAY_LEAD', 'LOSS'] if 'AWAY_LEAD' in ht_ft_matrix.index else 0,
                    'draw_stability': ht_ft_matrix.loc['DRAW', 'DRAW'] if 'DRAW' in ht_ft_matrix.index else 0
                }
            }

            # Goal scoring patterns
            goal_df = df.copy()
            goal_df['total_goals'] = goal_df['total_goals'].fillna(0)  # Handle NaN values
            goal_df['goals_range'] = pd.cut(goal_df['total_goals'],
                                            bins=[0, 1, 2, 3, 4, float('inf')],
                                            labels=['0-1', '1-2', '2-3', '3-4', '4+'],
                                            include_lowest=True)

            for dc_market in ['dc_1X', 'dc_X2', 'dc_12']:
                patterns['goal_patterns'][dc_market] = {}
                for goals, group in goal_df.groupby('goals_range', observed=True):
                    if pd.notna(goals):
                        patterns['goal_patterns'][dc_market][str(goals)] = group[dc_market].mean()

            # Competition patterns
            for comp_type in df['competition_type'].unique():
                comp_df = df[df['competition_type'] == comp_type]
                if len(comp_df) > 0:
                    patterns['competition_patterns'][comp_type] = {
                        'success_rates': {
                            '1X': comp_df['dc_1X'].mean(),
                            'X2': comp_df['dc_X2'].mean(),
                            '12': comp_df['dc_12'].mean()
                        },
                        'sample_size': len(comp_df)
                    }

            # Seasonal patterns
            monthly_patterns = df.groupby('month').agg({
                'dc_1X': 'mean',
                'dc_X2': 'mean',
                'dc_12': 'mean'
            }).to_dict()

            yearly_patterns = df.groupby('year').agg({
                'dc_1X': 'mean',
                'dc_X2': 'mean',
                'dc_12': 'mean'
            }).to_dict()

            patterns['seasonal_patterns'] = {
                'monthly': monthly_patterns,
                'yearly': yearly_patterns
            }

            # Score patterns
            df['team1_score_ft'] = df['team1_score_ft'].fillna(0)
            df['team2_score_ft'] = df['team2_score_ft'].fillna(0)
            common_scores = pd.crosstab(df['team1_score_ft'], df['team2_score_ft'])
            patterns['score_patterns'] = {
                'common_scores': common_scores.to_dict(),
                'dc_success_by_margin': {}
            }

            # Calculate success rates by goal margin
            df['goal_margin'] = abs(df['team1_score_ft'] - df['team2_score_ft'])
            for margin in sorted(df['goal_margin'].unique()):
                if pd.notna(margin):
                    margin_df = df[df['goal_margin'] == margin]
                    patterns['score_patterns']['dc_success_by_margin'][int(margin)] = {
                        '1X': margin_df['dc_1X'].mean(),
                        'X2': margin_df['dc_X2'].mean(),
                        '12': margin_df['dc_12'].mean()
                    }

            # Additional insights
            patterns['key_insights'] = {
                'best_conditions': {
                    '1X': {
                        'overall_rate': df['dc_1X'].mean(),
                        'best_competition': max(patterns['competition_patterns'].items(),
                                                key=lambda x: x[1]['success_rates']['1X'])[0],
                        'best_goals_range': max(
                            ((k, v) for k, v in patterns['goal_patterns']['dc_1X'].items() if pd.notna(v)),
                            key=lambda x: x[1])[0] if patterns['goal_patterns']['dc_1X'] else 'N/A'
                    },
                    'X2': {
                        'overall_rate': df['dc_X2'].mean(),
                        'best_competition': max(patterns['competition_patterns'].items(),
                                                key=lambda x: x[1]['success_rates']['X2'])[0],
                        'best_goals_range': max(
                            ((k, v) for k, v in patterns['goal_patterns']['dc_X2'].items() if pd.notna(v)),
                            key=lambda x: x[1])[0] if patterns['goal_patterns']['dc_X2'] else 'N/A'
                    },
                    '12': {
                        'overall_rate': df['dc_12'].mean(),
                        'best_competition': max(patterns['competition_patterns'].items(),
                                                key=lambda x: x[1]['success_rates']['12'])[0],
                        'best_goals_range': max(
                            ((k, v) for k, v in patterns['goal_patterns']['dc_12'].items() if pd.notna(v)),
                            key=lambda x: x[1])[0] if patterns['goal_patterns']['dc_12'] else 'N/A'
                    }
                }
            }

            return patterns

        except Exception as e:
            print(f"Error analyzing patterns: {str(e)}")
            return {}

    def create_visualizations(self, df: pd.DataFrame, probabilities: Dict, patterns: Dict):
        """Create comprehensive visualizations for Double Chance analysis"""
        if df.empty:
            print("No data available for visualizations")
            return None

        try:
            import plotly.graph_objects as go

            # Create subplots in a 3x3 grid
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    'Overall Double Chance Probabilities',
                    'Success Rates by Half-Time Result',
                    'Success by Competition Type',
                    'Success Rates by Goal Range',
                    'Monthly Trends',
                    'Success by Goal Margin',
                    'HT-FT Relationship',
                    'Competition Distribution',
                    'Success Rate Correlations'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}, {"type": "heatmap"}],
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                    [{"type": "heatmap"}, {"type": "pie"}, {"type": "heatmap"}]
                ]
            )

            # 1. Overall Probabilities (Row 1, Col 1)
            overall_probs = pd.DataFrame({
                'Market': ['1X', 'X2', '12'],
                'Probability': [
                    probabilities['overall']['1X'],
                    probabilities['overall']['X2'],
                    probabilities['overall']['12']
                ]
            })

            fig.add_trace(
                go.Bar(
                    x=overall_probs['Market'],
                    y=overall_probs['Probability'],
                    text=[f"{x:.1%}" for x in overall_probs['Probability']],
                    textposition='auto',
                    name='Overall Probability',
                    marker_color=['#2ecc71', '#3498db', '#e74c3c']
                ),
                row=1, col=1
            )

            # 2. HT Result Impact (Row 1, Col 2)
            if 'by_ht_result' in probabilities:
                ht_probs = pd.DataFrame(probabilities['by_ht_result']).T
                for market in ['1X', 'X2', '12']:
                    fig.add_trace(
                        go.Bar(
                            name=market,
                            x=ht_probs.index,
                            y=ht_probs[market],
                            text=[f"{x:.1%}" for x in ht_probs[market]],
                            textposition='auto'
                        ),
                        row=1, col=2
                    )

            # 3. Competition Type Heatmap (Row 1, Col 3)
            if 'competition_patterns' in patterns:
                comp_data = pd.DataFrame({
                    comp: data['success_rates']
                    for comp, data in patterns['competition_patterns'].items()
                }).T

                fig.add_trace(
                    go.Heatmap(
                        z=comp_data.values,
                        x=['1X', 'X2', '12'],
                        y=comp_data.index,
                        text=np.round(comp_data.values, 3),
                        texttemplate='%{text:.1%}',
                        textfont={"size": 10},
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    row=1, col=3
                )

            # 4. Goals Range Analysis (Row 2, Col 1)
            if 'by_total_goals' in probabilities:
                goals_df = pd.DataFrame(probabilities['by_total_goals']).T
                for market in ['1X', 'X2', '12']:
                    fig.add_trace(
                        go.Bar(
                            name=market,
                            x=goals_df.index,
                            y=goals_df[market],
                            text=[f"{x:.1%}" for x in goals_df[market]],
                            textposition='auto'
                        ),
                        row=2, col=1
                    )

            # 5. Monthly Trends (Row 2, Col 2)
            if 'seasonal_patterns' in patterns and 'monthly' in patterns['seasonal_patterns']:
                monthly_data = pd.DataFrame(patterns['seasonal_patterns']['monthly'])
                for market in ['dc_1X', 'dc_X2', 'dc_12']:
                    fig.add_trace(
                        go.Scatter(
                            x=monthly_data.index,
                            y=monthly_data[market],
                            name=market.replace('dc_', ''),
                            mode='lines+markers'
                        ),
                        row=2, col=2
                    )

            # 6. Goal Margin Success (Row 2, Col 3)
            if 'score_patterns' in patterns and 'dc_success_by_margin' in patterns['score_patterns']:
                margin_data = pd.DataFrame(patterns['score_patterns']['dc_success_by_margin']).T
                for market in ['1X', 'X2', '12']:
                    fig.add_trace(
                        go.Bar(
                            name=market,
                            x=margin_data.index,
                            y=margin_data[market],
                            text=[f"{x:.1%}" for x in margin_data[market]],
                            textposition='auto'
                        ),
                        row=2, col=3
                    )

            # 7. HT-FT Relationship (Row 3, Col 1)
            if 'ht_ft_patterns' in patterns and 'matrix' in patterns['ht_ft_patterns']:
                ht_ft_matrix = pd.DataFrame(patterns['ht_ft_patterns']['matrix'])
                fig.add_trace(
                    go.Heatmap(
                        z=ht_ft_matrix.values,
                        x=ht_ft_matrix.columns,
                        y=ht_ft_matrix.index,
                        text=np.round(ht_ft_matrix.values, 3),
                        texttemplate='%{text:.1%}',
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    row=3, col=1
                )

            # 8. Competition Distribution (Row 3, Col 2)
            comp_dist = df['competition'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=comp_dist.index,
                    values=comp_dist.values,
                    textinfo='percent+label',
                    hole=0.3
                ),
                row=3, col=2
            )

            # 9. Correlation Matrix (Row 3, Col 3)
            dc_cols = ['dc_1X', 'dc_X2', 'dc_12']
            corr_matrix = df[dc_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=dc_cols,
                    y=dc_cols,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text:.2f}',
                    colorscale='RdBu',
                    showscale=True
                ),
                row=3, col=3
            )

            # Update layout with valid properties
            fig.update_layout(
                height=1500,
                width=1800,
                title={
                    'text': "Double Chance Market Analysis Dashboard",
                    'x': 0.5,
                    'y': 0.98,
                    'font': {'size': 24}
                },
                showlegend=True,
                template="plotly_white",
                legend={
                    'orientation': "h",
                    'yanchor': "bottom",
                    'y': 1.02,
                    'xanchor': "right",
                    'x': 1
                },
                font={'size': 10}
            )

            # Update axes labels and formatting
            # Row 1
            fig.update_yaxes(title_text="Probability", row=1, col=1)
            fig.update_yaxes(title_text="Success Rate", row=1, col=2)
            fig.update_xaxes(title_text="Market", row=1, col=1)
            fig.update_xaxes(title_text="Half-Time Result", row=1, col=2)

            # Row 2
            fig.update_yaxes(title_text="Success Rate", row=2, col=1)
            fig.update_yaxes(title_text="Success Rate", row=2, col=2)
            fig.update_xaxes(title_text="Goals Range", row=2, col=1)
            fig.update_xaxes(title_text="Month", row=2, col=2)
            fig.update_xaxes(title_text="Goal Margin", row=2, col=3)

            # Add grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            # Update spacing
            fig.update_layout(
                height=1500,
                width=1800,
                bargap=0.2,
                bargroupgap=0.1
            )

            return fig

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    def print_insights(self, probabilities: Dict, patterns: Dict):
        """Print key insights from the analysis"""
        try:
            print("\n=== Double Chance Market Insights ===")

            if 'overall' in probabilities:
                print("\nOverall Probabilities:")
                for market, prob in probabilities['overall'].items():
                    print(f"{market}: {prob:.1%}")

            if 'key_insights' in patterns:
                print("\nBest Conditions by Market:")
                for market, insights in patterns['key_insights']['best_conditions'].items():
                    print(f"\n{market}:")
                    print(f"Overall Success Rate: {insights['overall_rate']:.1%}")
                    print(f"Best in Competition: {insights['best_competition']}")
                    print(f"Optimal Goals Range: {insights['best_goals_range']}")

            if 'ht_ft_patterns' in patterns and 'insights' in patterns['ht_ft_patterns']:
                print("\nHT-FT Conversion Rates:")
                insights = patterns['ht_ft_patterns']['insights']
                print(f"Home Lead Conversion: {insights.get('home_lead_conversion', 0):.1%}")
                print(f"Away Lead Conversion: {insights.get('away_lead_conversion', 0):.1%}")
                print(f"Draw Stability: {insights.get('draw_stability', 0):.1%}")

            if 'score_patterns' in patterns and 'dc_success_by_margin' in patterns['score_patterns']:
                print("\nSuccess by Goal Margin:")
                for margin, rates in patterns['score_patterns']['dc_success_by_margin'].items():
                    print(f"\nMargin of {margin} goal(s):")
                    for market, rate in rates.items():
                        print(f"{market}: {rate:.1%}")

            if 'seasonal_patterns' in patterns and 'monthly' in patterns['seasonal_patterns']:
                print("\nSeasonal Trends:")
                monthly_data = patterns['seasonal_patterns']['monthly']
                if 'dc_1X' in monthly_data:
                    best_month = max(monthly_data['dc_1X'].items(), key=lambda x: x[1])[0]
                    print(f"Best month for 1X: {best_month}")

        except Exception as e:
            print(f"Error printing insights: {str(e)}")

def main():
    try:
        # Initialize analyzer
        print("Initializing Double Chance Analyzer...")
        analyzer = DoubleChanceAnalyzer('MatchMind_db.db')

        # Load data
        print("Loading match data...")
        df = analyzer.load_data()

        if df.empty:
            print("Error: No data available for analysis")
            return

        print(f"Loaded {len(df)} matches for analysis")

        # Calculate probabilities
        print("Calculating probabilities...")
        probabilities = analyzer.calculate_dc_probabilities(df)

        if not probabilities:
            print("Error: Could not calculate probabilities")
            return

        # Analyze patterns
        print("Analyzing patterns...")
        patterns = analyzer.analyze_patterns(df)

        # Create visualizations
        print("Creating visualizations...")
        fig = analyzer.create_visualizations(df, probabilities, patterns)
        fig.show()

        # Print insights
        print("Generating insights...")
        analyzer.print_insights(probabilities, patterns)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
