import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import timeit
from google.generativeai import GenerativeModel, configure
from scipy import stats
from sklearn.decomposition import PCA

# Replace with your API key
GOOGLE_API_KEY = "AIzaSyA5HtRnzGruiia-aKtMMLnBjJ0ovTh11nE"

class Tools:
    def __init__(self):
        self.available_tools = {
            "plotly_scatter_plot": self.plotly_scatter_plot,
            "plotly_line_chart": self.plotly_line_chart,
            "plotly_bar_chart": self.plotly_bar_chart,
            "plotly_3d_surface": self.plotly_3d_surface,
            "plotly_pie_chart": self.plotly_pie_chart,
            "plotly_histogram": self.plotly_histogram,
            "plotly_3d_scatter": self.plotly_3d_scatter,
            "plotly_animated_chart": self.plotly_animated_chart,
            "plotly_heatmap": self.plotly_heatmap,
            "plotly_box_plot": self.plotly_box_plot,
            "plotly_violin_plot": self.plotly_violin_plot,
            "plotly_parallel_coordinates": self.plotly_parallel_coordinates,
            "plotly_scatter_matrix": self.plotly_scatter_matrix,
            "statistical_validation": self.statistical_validation,
            "plot_correlation_heatmap": self.plot_correlation_heatmap,
            "plot_dimensionality_reduction": self.plot_dimensionality_reduction
        }

    def plotly_scatter_plot(self, df, x, y, color=None, title="Scatter Plot"):
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig)

    def plotly_line_chart(self, df, x, y, color=None, title="Line Chart"):
        fig = px.line(df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig)

    def plotly_bar_chart(self, df, x, y, color=None, title="Bar Chart"):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig)

    def plotly_3d_surface(self, df, x, y, z, title="3D Surface Plot"):
        fig = go.Figure(data=[go.Surface(z=df[z].values.reshape(len(df[x].unique()), len(df[y].unique())),
                                         x=df[x].unique(), y=df[y].unique())])
        fig.update_layout(title=title, autosize=False, width=800, height=600)
        st.plotly_chart(fig)

    def plotly_pie_chart(self, df, names, values, title="Pie Chart"):
        fig = px.pie(df, names=names, values=values, title=title)
        st.plotly_chart(fig)

    def plotly_histogram(self, df, x, color=None, title="Histogram"):
        fig = px.histogram(df, x=x, color=color, title=title)
        st.plotly_chart(fig)

    def plotly_3d_scatter(self, df, x, y, z, color=None, title="3D Scatter Plot"):
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title)
        st.plotly_chart(fig)

    def plotly_animated_chart(self, df, x, y, animation_frame, title="Animated Chart"):
        fig = px.scatter(df, x=x, y=y, animation_frame=animation_frame, title=title)
        st.plotly_chart(fig)

    def plotly_heatmap(self, df, title="Heatmap"):
        fig = px.imshow(df.corr(), title=title)
        st.plotly_chart(fig)

    def plotly_box_plot(self, df, x, y, color=None, title="Box Plot"):
        fig = px.box(df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig)

    def plotly_violin_plot(self, df, x, y, color=None, title="Violin Plot"):
        fig = px.violin(df, x=x, y=y, color=color, box=True, points="all", title=title)
        st.plotly_chart(fig)

    def plotly_parallel_coordinates(self, df, color, title="Parallel Coordinates Plot"):
        fig = px.parallel_coordinates(df, color=color, title=title)
        st.plotly_chart(fig)

    def plotly_scatter_matrix(self, df, dimensions, color, title="Scatter Matrix"):
        fig = px.scatter_matrix(df, dimensions=dimensions, color=color, title=title)
        st.plotly_chart(fig)

    def statistical_validation(self, data1, data2):
        t_stat, p_value = stats.ttest_ind(data1, data2)
        results = {'t-statistic': t_stat, 'p-value': p_value}
        fig = go.Figure(data=[go.Bar(x=list(results.keys()), y=list(results.values()))])
        fig.update_layout(title="Statistical Test Results")
        st.plotly_chart(fig)
        return results

    def plot_correlation_heatmap(self, df):
        corr = df.corr()
        fig = px.imshow(corr, title="Feature Correlation Heatmap")
        st.plotly_chart(fig)

    def plot_dimensionality_reduction(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = StandardScaler().fit_transform(df[numeric_cols])
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        fig = px.scatter(x=components[:, 0], y=components[:, 1],
                         title="PCA Visualization",
                         labels={'x': 'First Principal Component',
                                 'y': 'Second Principal Component'})
        st.plotly_chart(fig)
        explained_variance = pca.explained_variance_ratio_
        st.write(f"Explained variance ratio: {explained_variance}")

def analyze_and_validate_data(df, experiment_type, tools):
    try:
        # Generate insights using Gemini API
        prompt = f"Analyze this dataset from a {experiment_type} biological experiment and provide insights:\n{df.head(20).to_string()}"
        model = GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        st.write(f"Gemini AI Summary for {experiment_type} experiment:")
        st.write(response.text)

        # Data Validation Section
        st.write("Data Validation and Quality Metrics:")
        
        # Basic data quality checks
        st.write("1. Data Quality Summary:")
        st.write(f"- Missing values: {df.isnull().sum().sum()}")
        st.write(f"- Duplicate rows: {df.duplicated().sum()}")
        st.write(f"- Number of features: {df.shape[1]}")
        st.write(f"- Number of samples: {df.shape[0]}")

        # Correlation analysis
        st.write("\n2. Feature Correlation Analysis:")
        tools.plot_correlation_heatmap(df)

        # Distribution analysis
        st.write("\n3. Distribution Analysis:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            tools.plotly_box_plot(df, y=col, title=f"Box Plot: {col}")
            tools.plotly_violin_plot(df, y=col, title=f"Violin Plot: {col}")
            tools.plotly_histogram(df, x=col, title=f"Histogram: {col}")

        # Dimensionality reduction
        st.write("\n4. Dimensionality Reduction Analysis:")
        tools.plot_dimensionality_reduction(df)

        # Statistical comparison (if earth data is available)
        if experiment_type == "space" and "earth_experiment_data" in st.session_state:
            st.write("\n5. Statistical Comparison with Earth Data:")
            earth_df = st.session_state["earth_experiment_data"]
            
            for col in df.select_dtypes(include=[np.number]).columns:
                st.write(f"\nAnalysis for {col}:")
                tools.statistical_validation(df[col], earth_df[col])

        # Advanced visualizations
        st.write("\nAdvanced Visualizations:")
        
        # Scatter plot matrix
        st.write("Scatter Plot Matrix:")
        tools.plotly_scatter_matrix(df, dimensions=numeric_cols[:4], color=df.columns[0], title="Scatter Plot Matrix")
        
        # Parallel coordinates plot
        st.write("Parallel Coordinates Plot:")
        tools.plotly_parallel_coordinates(df, color=df.columns[0], title="Parallel Coordinates Plot")
        
        # 3D scatter plot (if at least 3 numeric columns)
        if len(numeric_cols) >= 3:
            st.write("3D Scatter Plot:")
            tools.plotly_3d_scatter(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], color=df.columns[0], title="3D Scatter Plot")
        
        # Animated chart (if time-related column exists)
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            st.write("Animated Chart:")
            tools.plotly_animated_chart(df, x=numeric_cols[0], y=numeric_cols[1], animation_frame=time_cols[0], title="Animated Chart")

        # Space vs Earth comparison (if applicable)
        if experiment_type == "space" and "earth_experiment_data" in st.session_state:
            st.write("\nSpace vs Earth Comparisons:")
            earth_df = st.session_state["earth_experiment_data"]
            
            combined_df = pd.concat([
                df.assign(experiment_type='Space'),
                earth_df.assign(experiment_type='Earth')
            ])
            
            for col in numeric_cols:
                tools.plotly_box_plot(combined_df, x='experiment_type', y=col, title=f"Box Plot: {col} (Space vs Earth)")
                tools.plotly_violin_plot(combined_df, x='experiment_type', y=col, title=f"Violin Plot: {col} (Space vs Earth)")

    except Exception as e:
        st.write(f"An error occurred during analysis: {e}")

def main():
    st.title("Enhanced Space vs. Earth Biological Data Visualization")
    st.write("""
    This application provides comprehensive analysis and visualization of biological 
    experiments conducted in space and on Earth, including statistical validation 
    and performance metrics.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())

            if not GOOGLE_API_KEY:
                st.write("Please set your GOOGLE_API_KEY environment variable.")
            else:
                configure(api_key=GOOGLE_API_KEY)

                experiment_type = st.selectbox("Select the experiment type", 
                                             ["space", "earth"])

                if experiment_type == "earth":
                    st.session_state["earth_experiment_data"] = df

                tools = Tools()
                analyze_and_validate_data(df, experiment_type, tools)

        except Exception as e:
            st.write(f"An error occurred while loading the file: {e}")

if __name__ == "__main__":
    main()