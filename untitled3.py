import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('HR-Employee-Attrition.csv')

# Create a new column 'AttritionNumeric' with numerical values
df['AttritionNumeric'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Pair Plot for Feature Distribution
features_for_pair_plot = [
    'Age', 'DailyRate', 'HourlyRate', 'MonthlyIncome', 'TotalWorkingYears', 'AttritionNumeric'
]
pair_plot = px.scatter_matrix(
    df[features_for_pair_plot],
    dimensions=features_for_pair_plot,
    color='AttritionNumeric',
    title='Pair Plot of Selected Features',
    color_continuous_scale=["#07F49D", "#42047E"]  # Updated color codes
)
pair_plot.write_html('pair_plot_dataset.html')

# Count plot for Attrition
count_plot = px.bar(
    df,
    x='Attrition',
    color='Attrition',
    title='Attrition Count',
    labels={'x': 'Attrition', 'y': 'Count'},
    color_continuous_scale=["#07F49D", "#42047E"],  # Updated color codes
    barmode='overlay',  # Use 'overlay' for transparency
)
count_plot.write_html('count_plot_dataset.html')

# Histogram for Age distribution
age_histogram = px.histogram(
    df,
    x='Age',
    title='Age Distribution',
    labels={'Age': 'Age', 'count': 'Count'},
    color_discrete_sequence=["#07F49D"]  # Updated color code
)
age_histogram.write_html('age_histogram.html')

# Bar chart for BusinessTravel distribution
business_travel_chart = px.bar(
    df,
    x='BusinessTravel',
    title='Business Travel Distribution',
    labels={'BusinessTravel': 'Business Travel', 'count': 'Count'},
    color_discrete_sequence=["#42047E"]  # Updated color code
)
business_travel_chart.update_layout(xaxis={'categoryorder':'total descending'})
business_travel_chart.write_html('business_travel_chart.html')

# Filter only numeric columns for correlation matrix
numeric_columns = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_columns].corr()

# Correlation Heatmap
heatmap = px.imshow(
    correlation_matrix.values,
    labels=dict(color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    color_continuous_scale=["#07F49D", "#42047E"]  # Updated color codes
)
heatmap.write_html('heatmap_dataset.html')

# Box Plot for MonthlyIncome distribution based on Attrition
box_plot = px.box(
    df,
    x='Attrition',
    y='MonthlyIncome',
    color='Attrition',
    title='MonthlyIncome Distribution based on Attrition',
    labels={'Attrition': 'Attrition', 'MonthlyIncome': 'Monthly Income'},
color_discrete_map={'No': '#07F49D', 'Yes': '#42047E'} , # Updated color codes
)
box_plot.write_html('box_plot_dataset.html')
