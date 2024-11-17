import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketAnalysis:
    name: str
    query: str
    features: List[str]
    description: str


class BettingMarketAnalyzer:
    def __init__(self, db_path: str = 'MatchMind_db.db'):
        self.db_path = db_path
        self.markets = self._initialize_markets()

    def _initialize_markets(self) -> Dict[str, MarketAnalysis]:
        """Initialize market definitions and their analysis parameters"""
        return {
            '1X2': MarketAnalysis(
                name='Match Result (1X2)',
                query="""
                    SELECT 
                        competition, season, date,
                        team1, team2, ft_result,
                        team1_score_ht, team2_score_ht,
                        team1_score_ft, team2_score_ft,
                        total_goals, goals_per_half,
                        competition_type, country
                    FROM matches 
                    WHERE match_status = 'COMPLETED'
                """,
                features=['ft_result', 'total_goals', 'goals_per_half'],
                description='Home/Draw/Away market analysis'
            ),
            'BTTS': MarketAnalysis(
                name='Both Teams to Score',
                query="""
                    SELECT 
                        competition, season, date,
                        team1, team2, btts,
                        team1_score_ht, team2_score_ht,
                        clean_sheet_team1, clean_sheet_team2,
                        competition_type, country
                    FROM matches 
                    WHERE match_status = 'COMPLETED'
                """,
                features=['btts', 'clean_sheet_team1', 'clean_sheet_team2'],
                description='Both teams to score analysis'
            ),
            'OVER_UNDER': MarketAnalysis(
                name='Over/Under Goals',
                query="""
                    SELECT 
                        competition, season, date,
                        team1, team2, total_goals,
                        over_0_5, over_1_5, over_2_5,
                        over_3_5, over_4_5, over_5_5,
                        under_0_5, under_1_5, under_2_5,
                        under_3_5, under_4_5, under_5_5,
                        competition_type, country
                    FROM matches 
                    WHERE match_status = 'COMPLETED'
                """,
                features=['total_goals', 'over_2_5', 'under_2_5'],
                description='Over/Under goals market analysis'
            )
        }

    def load_market_data(self, market_key: str) -> pd.DataFrame:
        """Load data for specific market analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            market = self.markets[market_key]
            df = pd.read_sql_query(market.query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading {market_key} data: {str(e)}")
            return pd.DataFrame()

    def create_1x2_visualizations(self):
        """Create visualizations for 1X2 market analysis"""
        df = self.load_market_data('1X2')

        # Create result distribution DataFrame
        result_dist = df['ft_result'].value_counts().reset_index()
        result_dist.columns = ['Result', 'Count']

        # Overall distribution plot
        fig1 = px.bar(
            result_dist,
            x='Result', y='Count',
            title='Overall Match Result Distribution',
            color='Result',
            labels={'Count': 'Number of Matches'}
        )

        # Time series analysis by season
        season_dist = df.groupby(['season', 'ft_result']).size().reset_index(name='Count')
        fig2 = px.line(
            season_dist,
            x='season', y='Count',
            color='ft_result',
            title='Match Result Trends by Season',
            labels={'ft_result': 'Result', 'Count': 'Number of Matches'}
        )

        # Score pattern heatmap
        score_pattern = pd.crosstab(df['team1_score_ft'], df['team2_score_ft'])
        fig3 = px.imshow(
            score_pattern,
            title='Score Pattern Heatmap',
            labels=dict(x='Away Goals', y='Home Goals', color='Frequency')
        )

        return [fig1, fig2, fig3]

    def create_btts_visualizations(self):
        """Create visualizations for BTTS market analysis"""
        df = self.load_market_data('BTTS')

        # BTTS rate by competition
        btts_by_comp = df.groupby('competition')['btts'].agg(['mean', 'count']).reset_index()
        btts_by_comp.columns = ['Competition', 'BTTS_Rate', 'Match_Count']
        btts_by_comp = btts_by_comp.sort_values('BTTS_Rate', ascending=False)

        fig1 = px.bar(
            btts_by_comp,
            x='Competition', y='BTTS_Rate',
            title='BTTS Rate by Competition',
            labels={'BTTS_Rate': 'BTTS Rate'},
            hover_data=['Match_Count']
        )

        # BTTS trends over time
        df['date'] = pd.to_datetime(df['date'])
        btts_trend = df.set_index('date').groupby(pd.Grouper(freq='M'))['btts'].mean().reset_index()

        fig2 = px.line(
            btts_trend,
            x='date', y='btts',
            title='Monthly BTTS Rate Trend',
            labels={'btts': 'BTTS Rate', 'date': 'Month'}
        )

        # Clean sheet correlation
        clean_sheet_data = df.groupby('competition').agg({
            'clean_sheet_team1': 'mean',
            'clean_sheet_team2': 'mean',
            'btts': 'mean'
        }).reset_index()

        fig3 = px.scatter(
            clean_sheet_data,
            x='clean_sheet_team1',
            y='btts',
            title='Clean Sheet vs BTTS Correlation',
            labels={
                'clean_sheet_team1': 'Home Clean Sheet Rate',
                'btts': 'BTTS Rate'
            },
            hover_data=['competition']
        )

        return [fig1, fig2, fig3]

    def create_over_under_visualizations(self):
        """Create visualizations for Over/Under market analysis"""
        df = self.load_market_data('OVER_UNDER')

        # Goals distribution
        fig1 = px.histogram(
            df,
            x='total_goals',
            nbins=10,
            title='Distribution of Total Goals per Match',
            labels={'total_goals': 'Goals', 'count': 'Frequency'}
        )

        # Over rates by competition
        over_rates = df.groupby('competition').agg({
            'over_2_5': 'mean',
            'total_goals': 'mean'
        }).reset_index()

        fig2 = px.bar(
            over_rates,
            x='competition',
            y=['over_2_5', 'total_goals'],
            title='Over 2.5 Goals Rate and Average Goals by Competition',
            barmode='group',
            labels={
                'value': 'Rate/Average',
                'competition': 'Competition',
                'variable': 'Metric'
            }
        )

        # Over/Under trends by season
        ou_trends = df.melt(
            id_vars=['season'],
            value_vars=['over_1_5', 'over_2_5', 'over_3_5'],
            var_name='market',
            value_name='rate'
        )
        ou_trends = ou_trends.groupby(['season', 'market'])['rate'].mean().reset_index()

        fig3 = px.line(
            ou_trends,
            x='season',
            y='rate',
            color='market',
            title='Over/Under Market Trends by Season',
            labels={
                'rate': 'Success Rate',
                'market': 'Market'
            }
        )

        return [fig1, fig2, fig3]

    def create_summary_dashboard(self):
        """Create a comprehensive dashboard of all markets"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '1X2 Distribution', 'BTTS Rate by Competition',
                'Goals Distribution', 'Over 2.5 Goals Rate',
                'Score Pattern Heatmap', 'Market Trends'
            ),
            vertical_spacing=0.12
        )

        # Load data for each market
        df_1x2 = self.load_market_data('1X2')
        df_btts = self.load_market_data('BTTS')
        df_ou = self.load_market_data('OVER_UNDER')

        # 1X2 Distribution
        result_dist = df_1x2['ft_result'].value_counts()
        fig.add_trace(
            go.Bar(
                x=result_dist.index,
                y=result_dist.values,
                name='Match Results'
            ),
            row=1, col=1
        )

        # BTTS Rate
        btts_rate = df_btts.groupby('competition')['btts'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=btts_rate.index,
                y=btts_rate.values,
                name='BTTS Rate'
            ),
            row=1, col=2
        )

        # Goals Distribution
        fig.add_trace(
            go.Histogram(
                x=df_ou['total_goals'],
                name='Goals Distribution'
            ),
            row=2, col=1
        )

        # Over 2.5 Rate
        over_rate = df_ou.groupby('competition')['over_2_5'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=over_rate.index,
                y=over_rate.values,
                name='Over 2.5 Rate'
            ),
            row=2, col=2
        )

        # Score Pattern Heatmap
        score_pattern = pd.crosstab(df_1x2['team1_score_ft'], df_1x2['team2_score_ft'])
        fig.add_trace(
            go.Heatmap(
                z=score_pattern.values,
                x=score_pattern.columns,
                y=score_pattern.index,
                colorscale='Viridis'
            ),
            row=3, col=1
        )

        # Market Trends
        df_ou['date'] = pd.to_datetime(df_ou['date'])
        monthly_trends = df_ou.set_index('date').groupby(pd.Grouper(freq='M')).agg({
            'over_2_5': 'mean',
            'total_goals': 'mean'
        }).reset_index()

        fig.add_trace(
            go.Scatter(
                x=monthly_trends['date'],
                y=monthly_trends['over_2_5'],
                name='Over 2.5',
                mode='lines'
            ),
            row=3, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_trends['date'],
                y=monthly_trends['total_goals'] / 3,  # Scaled for visualization
                name='Avg Goals/3',
                mode='lines'
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            showlegend=True,
            title_text="Football Betting Markets Analysis Dashboard",
            template="plotly_white"
        )

        return fig

    # Add these methods to the BettingMarketAnalyzer class

    def create_correlation_analysis(self):
        """Create comprehensive correlation analysis across all relevant features"""
        # Combine data from all markets
        df_1x2 = self.load_market_data('1X2')
        df_btts = self.load_market_data('BTTS')
        df_ou = self.load_market_data('OVER_UNDER')

        # Select numerical columns for correlation
        numerical_columns = [
            'team1_score_ht', 'team2_score_ht',
            'team1_score_ft', 'team2_score_ft',
            'total_goals', 'goals_per_half',
            'over_0_5', 'over_1_5', 'over_2_5',
            'over_3_5', 'over_4_5', 'over_5_5',
            'under_0_5', 'under_1_5', 'under_2_5',
            'under_3_5', 'under_4_5', 'under_5_5',
            'goal_difference', 'clean_sheet_team1',
            'clean_sheet_team2', 'btts'
        ]

        # Create combined DataFrame with all numerical features
        combined_df = pd.DataFrame()
        for col in numerical_columns:
            if col in df_1x2.columns:
                combined_df[col] = df_1x2[col]
            elif col in df_btts.columns:
                combined_df[col] = df_btts[col]
            elif col in df_ou.columns:
                combined_df[col] = df_ou[col]

        # Calculate correlation matrix
        corr_matrix = combined_df.corr()

        # Create correlation heatmap
        fig1 = px.imshow(
            corr_matrix,
            title='Feature Correlation Heatmap',
            labels=dict(color='Correlation'),
            color_continuous_scale='RdBu',
            aspect='auto'
        )

        # Find top correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlations.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': abs(corr_matrix.iloc[i, j])
                })

        # Sort correlations and get top 20
        top_correlations = sorted(correlations, key=lambda x: x['correlation'], reverse=True)[:20]

        # Create bar plot of top correlations
        top_corr_df = pd.DataFrame(top_correlations)
        fig2 = px.bar(
            top_corr_df,
            x='correlation',
            y=[f"{row['feature1']} vs {row['feature2']}" for _, row in top_corr_df.iterrows()],
            title='Top 20 Feature Correlations',
            orientation='h',
            labels={'y': 'Feature Pairs', 'x': 'Absolute Correlation'}
        )

        # Create scatter plots for top 5 correlations without trendline
        scatter_figs = []
        for i in range(5):
            feat1 = top_correlations[i]['feature1']
            feat2 = top_correlations[i]['feature2']

            fig = px.scatter(
                combined_df,
                x=feat1,
                y=feat2,
                title=f'Correlation: {feat1} vs {feat2} (r = {top_correlations[i]["correlation"]:.3f})',
                labels={
                    feat1: feat1.replace('_', ' ').title(),
                    feat2: feat2.replace('_', ' ').title()
                }
            )

            # Add custom layout options
            fig.update_traces(marker=dict(size=6, opacity=0.6))
            fig.update_layout(
                xaxis_title=feat1.replace('_', ' ').title(),
                yaxis_title=feat2.replace('_', ' ').title(),
                title_x=0.5,
                title_y=0.95
            )

            scatter_figs.append(fig)

        # Create detailed correlation analysis for each market type
        market_correlations = {}
        for market_key in ['1X2', 'BTTS', 'OVER_UNDER']:
            df = self.load_market_data(market_key)
            num_cols = df.select_dtypes(include=[np.number]).columns
            market_corr = df[num_cols].corr()

            fig = px.imshow(
                market_corr,
                title=f'{market_key} Market Correlation Heatmap',
                labels=dict(color='Correlation'),
                color_continuous_scale='RdBu',
                aspect='auto'
            )

            # Improve layout
            fig.update_layout(
                title_x=0.5,
                title_y=0.95,
                width=800,
                height=800
            )

            market_correlations[market_key] = fig

        # Calculate summary statistics
        summary_stats = {}
        for col in combined_df.columns:
            summary_stats[col] = {
                'mean': combined_df[col].mean(),
                'std': combined_df[col].std(),
                'min': combined_df[col].min(),
                'max': combined_df[col].max(),
                'null_count': combined_df[col].isnull().sum()
            }

        return {
            'overall_heatmap': fig1,
            'top_correlations': fig2,
            'top_5_scatter_plots': scatter_figs,
            'market_specific_correlations': market_correlations,
            'summary_statistics': summary_stats,
            'correlation_matrix': corr_matrix
        }

    def analyze_feature_importance(self):
        """Analyze feature importance for each market type with categorical handling"""
        importance_analysis = {}

        for market_key in self.markets.keys():
            df = self.load_market_data(market_key)
            target_var = self.markets[market_key].features[0]

            if target_var not in df.columns:
                continue

            # Handle categorical target variables
            if df[target_var].dtype == 'object':
                # Create dummy variables for categorical target
                target_dummies = pd.get_dummies(df[target_var], prefix=target_var)

                # Calculate importance for each category
                correlations = {}
                numeric_cols = df.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    # Calculate correlation with each target category
                    category_correlations = []
                    for dummy_col in target_dummies.columns:
                        corr = abs(df[col].corr(target_dummies[dummy_col]))
                        category_correlations.append(corr)
                    # Use the maximum correlation across categories
                    correlations[col] = max(category_correlations)

            else:
                # For numerical targets, proceed with regular correlation
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                correlations = {}
                for col in numeric_cols:
                    if col != target_var:
                        corr = abs(df[col].corr(df[target_var]))
                        correlations[col] = corr

            # Create feature importance plot
            if correlations:
                importance_df = pd.DataFrame({
                    'feature': list(correlations.keys()),
                    'importance': list(correlations.values())
                }).sort_values('importance', ascending=False)

                # Filter out NaN values
                importance_df = importance_df.dropna()

                # Create visualization
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    title=f'Feature Importance for {market_key} Market',
                    orientation='h',
                    labels={
                        'importance': 'Absolute Correlation',
                        'feature': 'Feature'
                    }
                )

                # Improve layout
                fig.update_layout(
                    title_x=0.5,
                    title_y=0.95,
                    width=800,
                    height=len(importance_df) * 30 + 100,  # Dynamic height
                    showlegend=False
                )

                # Calculate additional statistics
                feature_stats = {}
                for feature in importance_df['feature']:
                    feature_stats[feature] = {
                        'mean': df[feature].mean(),
                        'std': df[feature].std(),
                        'correlation': correlations[feature]
                    }

                importance_analysis[market_key] = {
                    'plot': fig,
                    'top_features': importance_df.head(5).to_dict('records'),
                    'all_features': correlations,
                    'feature_stats': feature_stats
                }

        return importance_analysis

    def create_categorical_correlation_analysis(self):
        """Create correlation analysis specifically for categorical variables"""
        categorical_analysis = {}

        for market_key in self.markets.keys():
            df = self.load_market_data(market_key)
            cat_cols = df.select_dtypes(include=['object']).columns
            num_cols = df.select_dtypes(include=[np.number]).columns

            market_analysis = {
                'categorical_distributions': {},
                'numerical_by_category': {}
            }

            # Analyze categorical distributions
            for cat_col in cat_cols:
                # Create distribution plot
                dist = df[cat_col].value_counts().reset_index()
                dist.columns = ['Category', 'Count']

                fig = px.bar(
                    dist,
                    x='Category',
                    y='Count',
                    title=f'Distribution of {cat_col}',
                    labels={'Count': 'Number of Occurrences'}
                )

                # Update layout
                fig.update_layout(
                    title_x=0.5,
                    width=800,
                    height=400
                )

                market_analysis['categorical_distributions'][cat_col] = {
                    'plot': fig,
                    'distribution': dist.to_dict('records')
                }

            # Analyze numerical variables by category
            for cat_col in cat_cols:
                category_stats = {}
                for num_col in num_cols:
                    stats_by_category = df.groupby(cat_col)[num_col].agg([
                        'mean', 'std', 'count'
                    ]).reset_index()

                    fig = px.bar(
                        stats_by_category,
                        x=cat_col,
                        y='mean',
                        error_y='std',
                        title=f'Mean {num_col} by {cat_col}',
                        labels={
                            'mean': f'Mean {num_col}',
                            cat_col: cat_col.replace('_', ' ').title()
                        }
                    )

                    # Update layout
                    fig.update_layout(
                        title_x=0.5,
                        width=800,
                        height=400
                    )

                    category_stats[num_col] = {
                        'plot': fig,
                        'stats': stats_by_category.to_dict('records')
                    }

                market_analysis['numerical_by_category'][cat_col] = category_stats

            categorical_analysis[market_key] = market_analysis

        return categorical_analysis

 # Example usage
if __name__ == "__main__":
        analyzer = BettingMarketAnalyzer('MatchMind_db.db')

        # Get both numerical and categorical analyses
        importance_results = analyzer.analyze_feature_importance()
        categorical_results = analyzer.create_categorical_correlation_analysis()

        # Display feature importance results
        for market, analysis in importance_results.items():
            print(f"\n=== {market} Market Analysis ===")
            print("\nTop 5 Important Features:")
            for feature in analysis['top_features']:
                stats = analysis['feature_stats'][feature['feature']]
                print(f"- {feature['feature']}:")
                print(f"  Importance: {feature['importance']:.3f}")
                print(f"  Mean: {stats['mean']:.3f}")
                print(f"  Std: {stats['std']:.3f}")

            # Display the plot
            analysis['plot'].show()

        # Display categorical analysis results
        for market, analysis in categorical_results.items():
            print(f"\n=== {market} Market Categorical Analysis ===")

            # Display categorical distributions
            for cat_col, dist_analysis in analysis['categorical_distributions'].items():
                print(f"\nDistribution of {cat_col}:")
                dist_analysis['plot'].show()

            # Display numerical statistics by category
            for cat_col, num_analysis in analysis['numerical_by_category'].items():
                print(f"\nNumerical Analysis by {cat_col}:")
                for num_col, stats in num_analysis.items():
                    stats['plot'].show()