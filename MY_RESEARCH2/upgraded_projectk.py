from sklearn.manifold import TSNE
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from datetime import datetime
import logging
from pathlib import Path
from google.generativeai import GenerativeModel, configure
import plotly.figure_factory as ff
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import umap
from sklearn.impute import KNNImputer
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GOOGLE_API_KEY = "AIzaSyA5HtRnzGruiia-aKtMMLnBjJ0ovTh11nE"
RANDOM_STATE = 42
VISUALIZATION_HEIGHT = 600
VISUALIZATION_WIDTH = 800

@dataclass
class AnalysisConfig:
    """Configuration parameters for analysis."""
    significance_level: float = 0.05
    cv_folds: int = 5
    test_size: float = 0.2
    outlier_threshold: float = 0.05
    min_samples_dbscan: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

class DataValidator:
    """Handles data validation and quality checks."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_results = {}
        
    def check_missing_values(self) -> Dict[str, float]:
        """Check for missing values in each column."""
        missing = self.df.isnull().sum() / len(self.df) * 100
        return missing[missing > 0].to_dict()
        
    def check_duplicates(self) -> int:
        """Check for duplicate rows."""
        return self.df.duplicated().sum()
        
    def check_outliers(self, method: str = 'zscore') -> Dict[str, np.ndarray]:
        """Detect outliers using various methods."""
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
            outliers = {col: np.where(z_scores[:, i] > 3)[0] 
                       for i, col in enumerate(numeric_cols)}
        elif method == 'isolation_forest':
            iso = IsolationForest(random_state=RANDOM_STATE)
            outliers = {col: np.where(iso.fit_predict(
                self.df[numeric_cols]) == -1)[0] for col in numeric_cols}
        
        return outliers
        
    def check_distributions(self) -> Dict[str, Dict[str, float]]:
        """Check normality and other distribution properties."""
        results = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stat, p_value = shapiro(self.df[col])
            results[col] = {
                'shapiro_stat': stat,
                'shapiro_p_value': p_value,
                'skewness': stats.skew(self.df[col]),
                'kurtosis': stats.kurtosis(self.df[col])
            }
        
        return results

class FeatureAnalyzer:
    """Analyzes features and their relationships."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    def compute_correlations(self) -> pd.DataFrame:
        """Compute correlation matrix."""
        return self.df[self.numeric_cols].corr()
        
    def feature_importance(self, target_col: str) -> pd.Series:
        """Calculate feature importance using Random Forest."""
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X, y)
        
        return pd.Series(rf.feature_importances_, index=X.columns)
        
    def mutual_information(self) -> pd.DataFrame:
        """Calculate mutual information between features."""
        from sklearn.feature_selection import mutual_info_regression
        
        mi_scores = []
        for col in self.numeric_cols:
            mi = mutual_info_regression(
                self.df[self.numeric_cols].drop(columns=[col]),
                self.df[col]
            )
            mi_scores.append(mi)
        
        # Fix: Ensure we're creating a valid DataFrame    
        return pd.DataFrame(mi_scores, 
                            columns=self.numeric_cols.drop(col),
                            index=self.numeric_cols)
        
    

class DataVisualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def plot_missing_values(self) -> go.Figure:
        """Plot missing values heatmap."""
        missing = self.df.isnull()
        fig = px.imshow(missing.astype(int),
                       labels=dict(color="Missing"),
                       title="Missing Values Heatmap")
        return fig
        
    def plot_correlation_matrix(self) -> go.Figure:
        """Plot correlation matrix heatmap."""
        corr = self.df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig = px.imshow(np.ma.masked_array(corr, mask),
                       labels=dict(color="Correlation"),
                       title="Correlation Matrix")
        return fig
        
    def plot_feature_distributions(self) -> List[go.Figure]:
        """Plot distribution for each numeric feature."""
        figs = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            fig = ff.create_distplot([self.df[col]], [col])
            fig.update_layout(title=f"Distribution of {col}")
            figs.append(fig)
            
        return figs
        
    def plot_dimensionality_reduction(self, 
                                    method: str = 'pca',
                                    n_components: int = 2) -> go.Figure:
        """Plot dimensionality reduction results."""
        numeric_data = self.df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components)
            
        reduced_data = reducer.fit_transform(scaled_data)
        
        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1],
                        title=f"{method.upper()} Visualization")
        return fig
        
    def plot_outliers(self, outliers: Dict[str, np.ndarray]) -> List[go.Figure]:
        """Plot outliers for each feature."""
        figs = []
        
        for col, indices in outliers.items():
            fig = go.Figure()
            
            # Regular points
            fig.add_trace(go.Scatter(
                x=np.arange(len(self.df)),
                y=self.df[col],
                mode='markers',
                name='Regular Points'
            ))
            
            # Outliers
            fig.add_trace(go.Scatter(
                x=indices,
                y=self.df[col].iloc[indices],
                mode='markers',
                name='Outliers',
                marker=dict(color='red')
            ))
            
            fig.update_layout(title=f"Outliers in {col}")
            figs.append(fig)
            
        return figs

class StatisticalAnalyzer:
    """Performs statistical analyses."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def compare_groups(self, 
                      group1: np.ndarray, 
                      group2: np.ndarray) -> Dict[str, Any]:
        """Perform statistical comparison between groups."""
        # Check normality
        _, p_norm1 = shapiro(group1)
        _, p_norm2 = shapiro(group2)
        
        # Check variance homogeneity
        _, p_var = levene(group1, group2)
        
        # Choose appropriate test
        if p_norm1 > self.config.significance_level and \
           p_norm2 > self.config.significance_level and \
           p_var > self.config.significance_level:
            # Use t-test
            stat, p_value = ttest_ind(group1, group2)
            test_name = "Student's t-test"
        else:
            # Use Mann-Whitney U test
            stat, p_value = mannwhitneyu(group1, group2)
            test_name = "Mann-Whitney U test"
            
        return {
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.config.significance_level
        }
        
    def perform_anova(self, 
                     groups: List[np.ndarray]) -> Dict[str, Any]:
        """Perform one-way ANOVA."""
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.config.significance_level
        }
        
    def compute_effect_size(self, 
                           group1: np.ndarray, 
                           group2: np.ndarray) -> Dict[str, float]:
        """Compute effect size metrics."""
        # Cohen's d
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Hedges' g
        n1, n2 = len(group1), len(group2)
        hedges_g = cohens_d * (1 - (3 / (4 * (n1 + n2) - 9)))
        
        return {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g
        }

class MachineLearningAnalyzer:
    """Performs machine learning analyses."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def train_classifier(self, 
                        X: pd.DataFrame, 
                        y: pd.Series) -> Dict[str, Any]:
        """Train and evaluate a classifier."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=RANDOM_STATE
        )
        
        # Train model
        clf = RandomForestClassifier(random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            clf, X, y, cv=self.config.cv_folds
        )
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        return {
            'model': clf,
            'cv_scores': cv_scores,
            'predictions': y_pred,
            'feature_importance': pd.Series(
                clf.feature_importances_, index=X.columns
            )
        }
        
    def perform_clustering(self, 
                         X: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis."""
        # Scale data
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # DBSCAN clustering
        dbscan = DBSCAN(
            min_samples=self.config.min_samples_dbscan
        )
        clusters = dbscan.fit_predict(X_scaled)
        
        return {
            'clusters': clusters,
            'n_clusters': len(np.unique(clusters[clusters != -1])),
            'noise_points': np.sum(clusters == -1)
        }
    

class ExperimentAnalyzer:
    """Main class for analyzing biological experiments."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def analyze_experiment(self, 
                          df: pd.DataFrame, 
                          experiment_type: str) -> Dict[str, Any]:
        """Perform comprehensive analysis on the experiment data."""
        try:
            results = {}
            
            # Data validation
            validator = DataValidator(df)
            results['validation'] = {
                'missing_values': validator.check_missing_values(),
                'duplicates': validator.check_duplicates(),
                'outliers': validator.check_outliers(),
                'distributions': validator.check_distributions()
            }
            
            # Feature analysis
            feature_analyzer = FeatureAnalyzer(df)
            results['feature_analysis'] = {
                'correlations': feature_analyzer.compute_correlations(),
                'mutual_information': feature_analyzer.mutual_information()
            }
            
            # Visualizations
            visualizer = DataVisualizer(df)
            results['visualizations'] = {
                'correlations': visualizer.plot_correlation_matrix(),
                'distributions': visualizer.plot_feature_distributions(),
                'pca': visualizer.plot_dimensionality_reduction(method='pca'),
                'umap': visualizer.plot_dimensionality_reduction(method='umap')
            }
            
            # Statistical analysis (if Earth experiment data is available)
            if experiment_type == 'space' and 'earth_experiment_data' in st.session_state:
                earth_df = st.session_state['earth_experiment_data']
                stat_analyzer = StatisticalAnalyzer(self.config)
                
                results['statistical_analysis'] = {}
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):  # Fix: Check for numeric type
                        results['statistical_analysis'][col] = {
                            'comparison': stat_analyzer.compare_groups(
                                df[col].values, earth_df[col].values
                            ),
                            'effect_size': stat_analyzer.compute_effect_size(
                                df[col].values, earth_df[col].values
                            )
                        }
                
                # Machine learning analysis
                ml_analyzer = MachineLearningAnalyzer(self.config)
                
                # Prepare combined dataset for ML
                combined_df = pd.concat([
                    df.assign(experiment_type='Space'),
                    earth_df.assign(experiment_type='Earth')
                ])
                
                X = combined_df.drop(columns=['experiment_type'])
                y = combined_df['experiment_type']
                
                results['ml_analysis'] = ml_analyzer.train_classifier(X, y)
                results['clustering'] = ml_analyzer.perform_clustering(X)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in experiment analysis: {str(e)}")
            raise

class ReportGenerator:
    """Generates comprehensive reports from analysis results."""
    
    def __init__(self, results: Dict[str, Any], experiment_type: str):
        self.results = results
        self.experiment_type = experiment_type
        
    def generate_markdown_report(self) -> str:
        """Generate a markdown report from the analysis results."""
        report = [
            "# Biological Experiment Analysis Report",
            f"## Experiment Type: {self.experiment_type}",
            f"### Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            
            "\n## 1. Data Quality Analysis",
            self._generate_data_quality_section(),
            
            "\n## 2. Feature Analysis",
            self._generate_feature_analysis_section(),
            
            "\n## 3. Statistical Analysis",
            self._generate_statistical_analysis_section(),
            
            "\n## 4. Machine Learning Insights",
            self._generate_ml_analysis_section(),
            
            "\n## 5. Recommendations",
            self._generate_recommendations()
        ]
        
        return "\n".join(report)
        
    def _generate_data_quality_section(self) -> str:
        """Generate the data quality section of the report."""
        validation = self.results['validation']
        
        sections = [
            "\n### Missing Values",
            f"Total missing values: {sum(validation['missing_values'].values())}",
            "\nBreakdown by column:",
            "\n".join([f"- {col}: {val:.2f}%" 
                      for col, val in validation['missing_values'].items()]),
            
            "\n### Duplicates",
            f"Total duplicate rows: {validation['duplicates']}",
            
            "\n### Outliers",
            "Outliers detected by column:",
            "\n".join([f"- {col}: {len(indices)} outliers" 
                      for col, indices in validation['outliers'].items()])
        ]
        
        return "\n".join(sections)
        
    def _generate_feature_analysis_section(self) -> str:
        """Generate the feature analysis section of the report."""
        feature_analysis = self.results['feature_analysis']
        
        sections = [
            "\n### Correlation Analysis",
            "Notable correlations (|r| > 0.7):",
            self._get_notable_correlations(feature_analysis['correlations']),
            
            "\n### Mutual Information",
            "Top feature relationships by mutual information:",
            self._get_top_mutual_information(feature_analysis['mutual_information'])
        ]
        
        return "\n".join(sections)
        
    def _generate_statistical_analysis_section(self) -> str:
        """Generate the statistical analysis section of the report."""
        if 'statistical_analysis' not in self.results:
            return "\nNo statistical analysis available for this experiment type."
            
        analysis = self.results['statistical_analysis']
        
        sections = ["\n### Statistical Tests"]
        for col, results in analysis.items():
            sections.extend([
                f"\n#### {col}",
                f"Test used: {results['comparison']['test_name']}",
                f"P-value: {results['comparison']['p_value']:.4f}",
                f"Significant: {'Yes' if results['comparison']['significant'] else 'No'}",
                "\nEffect Size:",
                f"- Cohen's d: {results['effect_size']['cohens_d']:.4f}",
                f"- Hedges' g: {results['effect_size']['hedges_g']:.4f}"
            ])
            
        return "\n".join(sections)
        
    def _generate_ml_analysis_section(self) -> str:
        """Generate the machine learning analysis section of the report."""
        if 'ml_analysis' not in self.results:
            return "\nNo machine learning analysis available for this experiment type."
            
        ml_analysis = self.results['ml_analysis']
        clustering = self.results['clustering']
        
        sections = [
            "\n### Classification Results",
            f"Cross-validation scores: {ml_analysis['cv_scores'].mean():.4f} "
            f"(±{ml_analysis['cv_scores'].std():.4f})",
            
            "\n### Feature Importance",
            "Top 5 most important features:",
            "\n".join([f"- {feat}: {imp:.4f}" for feat, imp in 
                      ml_analysis['feature_importance'].nlargest(5).items()]),
            
            "\n### Clustering Analysis",
            f"Number of clusters: {clustering['n_clusters']}",
            f"Noise points: {clustering['noise_points']}"
        ]
        
        return "\n".join(sections)
        
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on the analysis results."""
        recommendations = ["\nBased on the analysis, we recommend:"]
        
        # Data quality recommendations
        if self.results['validation']['missing_values']:
            recommendations.append(
                "- Consider implementing more robust data collection procedures "
                "to reduce missing values"
            )
            
        if self.results['validation']['duplicates'] > 0:
            recommendations.append(
                "- Investigate the source of duplicate entries and implement "
                "validation checks"
            )
            
        # Statistical recommendations
        if 'statistical_analysis' in self.results:
            significant_features = [
                col for col, results in 
                self.results['statistical_analysis'].items()
                if results['comparison']['significant']
            ]
            
            if significant_features:
                recommendations.append(
                    f"- Focus further investigation on the following features "
                    f"that showed significant differences: {', '.join(significant_features)}"
                )
                
        return "\n".join(recommendations)
        
    @staticmethod
    def _get_notable_correlations(corr_matrix: pd.DataFrame) -> str:
        """Extract notable correlations from the correlation matrix."""
        notable = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    notable.append(
                        f"- {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: "
                        f"{corr_matrix.iloc[i, j]:.3f}"
                    )
        return "\n".join(notable)
        
    @staticmethod
    def _get_top_mutual_information(mi_matrix: pd.DataFrame) -> str:
        """Extract top mutual information relationships."""
        flat_mi = []
        for i in range(len(mi_matrix.columns)):
            for j in range(i + 1, len(mi_matrix.columns)):
                flat_mi.append((
                    mi_matrix.columns[i],
                    mi_matrix.columns[j],
                    mi_matrix.iloc[i, j]
                ))
                
        flat_mi.sort(key=lambda x: x[2], reverse=True)
        return "\n".join([
            f"- {pair[0]} vs {pair[1]}: {pair[2]:.3f}"
            for pair in flat_mi[:5]
        ])

def main():
    """Main application entry point."""
    st.set_page_config(page_title="Advanced Biological Data Analysis",
                      page_icon="🧬",
                      layout="wide")
    
    st.title("🧬 Advanced Space vs. Earth Biological Data Analysis")
    
    st.sidebar.header("Configuration")
    config = AnalysisConfig(
        significance_level=st.sidebar.slider(
            "Significance Level", 0.01, 0.10, 0.05, 0.01
        ),
        cv_folds=st.sidebar.slider("Cross-validation Folds", 2, 10, 5),
        test_size=st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05),
        outlier_threshold=st.sidebar.slider(
            "Outlier Threshold", 0.01, 0.10, 0.05, 0.01
        )
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Configure Google API
            if not GOOGLE_API_KEY:
                st.error("Please set your GOOGLE_API_KEY environment variable.")
                return
                
            configure(api_key=GOOGLE_API_KEY)
            
            # Select experiment type
            experiment_type = st.selectbox(
                "Select the experiment type",
                ["space", "earth"]
            )
            
            # Save earth experiment data for comparison
            if experiment_type == "earth":
                st.session_state["earth_experiment_data"] = df
            
            # Initialize analyzer
            analyzer = ExperimentAnalyzer(config)
            
            # Perform analysis
            with st.spinner("Analyzing data..."):
                results = analyzer.analyze_experiment(df, experiment_type)
            
            # Generate and display report
            report_generator = ReportGenerator(results, experiment_type)
            report = report_generator.generate_markdown_report()
            
            # Display results in tabs
            tabs = st.tabs([
                "Report", "Visualizations", "Statistical Analysis",
                "Machine Learning", "Raw Data"
            ])
            
            with tabs[0]:
                st.markdown(report)
                
            with tabs[1]:
                st.plotly_chart(
                    results['visualizations']['correlations'],
                    use_container_width=True
                )
                for fig in results['visualizations']['distributions']:
                    st.plotly_chart(fig, use_container_width=True)
                    
            with tabs[2]:
                if 'statistical_analysis' in results:
                    for col, analysis in results['statistical_analysis'].items():
                        st.write(f"### Analysis for {col}")
                        st.json(analysis)
                        
            with tabs[3]:
                if 'ml_analysis' in results:
                    st.write("### Feature Importance")
                    st.bar_chart(results['ml_analysis']['feature_importance'])
                    
            with tabs[4]:
                st.write("### Raw Data")
                st.dataframe(df)
            
            # Export options
            if st.button("Export Report"):
                report_path = Path(f"report_{experiment_type}_{datetime.now():%Y%m%d_%H%M%S}.md")
                report_path.write_text(report)
                st.success(f"Report exported to {report_path}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("Error in main application")

if __name__ == "__main__":
    main()