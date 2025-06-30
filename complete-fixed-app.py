# complete-fixed-app.py

import base64
import io
import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import firebase_admin
from firebase_admin import credentials, auth, firestore
import numpy as np
import pandas as pd
import pyrebase
import plotly.express as px
import xml.etree.ElementTree as ET
# complete-fixed-app.py
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
flask_app = Flask(__name__)
flask_app.secret_key = "your_secret_key" # You should also move this to an environment variable for production
bcrypt = Bcrypt(flask_app)

# MODIFIED: Load Firebase Admin credentials from an environment variable
# Get the JSON string from the environment variable
firebase_creds_json_str = os.environ.get('FIREBASE_CREDS_JSON')
if not firebase_creds_json_str:
    # This will raise an error during startup if the variable is not set
    raise ValueError("FIREBASE_CREDS_JSON environment variable not set.")

# Convert the JSON string into a dictionary
firebase_creds_dict = json.loads(firebase_creds_json_str)

# Initialize Firebase Admin SDK
cred = credentials.Certificate(firebase_creds_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# MODIFIED: Load Pyrebase config from environment variables as well
firebase_config = {
    "apiKey": os.environ.get("PYREBASE_API_KEY"),
    "authDomain": os.environ.get("PYREBASE_AUTH_DOMAIN"),
    "databaseURL": os.environ.get("PYREBASE_DATABASE_URL"),
    "projectId": os.environ.get("PYREBASE_PROJECT_ID"),
    "storageBucket": os.environ.get("PYREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.environ.get("PYREBASE_MESSAGING_SENDER_ID"),
    "appId": os.environ.get("PYREBASE_APP_ID")
}

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebase_config)
auth_client = firebase.auth()


# Dashboard class
class Dashboard:
    def __init__(self):
        self.app = Dash(
            __name__,
            server=flask_app,
            url_base_pathname="/dashboard/",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        # Store the current dataframe and filename
        self.df = None
        self.filename = None  # Store the original filename
        self.setup_layout()
        self.setup_callbacks()

    def generate_natural_language_summary(self, df, data_types):
        """Generates a dynamic, template-based summary of the dataset."""
        num_rows, num_cols = df.shape
        all_categorical = data_types['categorical_low'] + data_types['categorical_high']
        duplicate_rows = df.duplicated().sum()
        missing_values_total = df.isnull().sum().sum()

        summary_sentences = []
        summary_sentences.append(f"The dataset contains {num_rows:,} records and {num_cols} features.")

        quality_issues = []
        if duplicate_rows > 0:
            quality_issues.append(f"{duplicate_rows} duplicate rows")
        if missing_values_total > 0:
            quality_issues.append(f"{missing_values_total:,} missing values")

        if quality_issues:
            summary_sentences.append(f"In terms of data quality, there are { ' and '.join(quality_issues) } that may need attention.")
        else:
            summary_sentences.append("The dataset appears to be clean, with no duplicate rows or missing values detected.")

        composition_parts = []
        if data_types['numeric']: composition_parts.append(f"{len(data_types['numeric'])} numeric")
        if all_categorical: composition_parts.append(f"{len(all_categorical)} categorical")
        if data_types['datetime']: composition_parts.append(f"{len(data_types['datetime'])} date/time")
        if data_types['boolean']: composition_parts.append(f"{len(data_types['boolean'])} boolean")

        if composition_parts:
            summary_sentences.append(f"It is composed primarily of { ', '.join(composition_parts) } columns.")

        if data_types['datetime']:
            dt_col = data_types['datetime'][0]
            summary_sentences.append(f"The data seems to be time-sensitive, with '{dt_col}' available for time series analysis.")
        elif data_types['numeric'] and all_categorical:
            num_col = data_types['numeric'][0]
            cat_col = all_categorical[0]
            summary_sentences.append(f"Key features appear to include the numerical column '{num_col}' and the categorical column '{cat_col}', suggesting potential for comparative analysis.")
        elif data_types['numeric']:
            summary_sentences.append(f"The dataset is largely numerical, making it suitable for statistical and correlation analysis.")

        return " ".join(summary_sentences)

    def generate_dataset_summary(self, df):
        """Generate comprehensive dataset summary, now with a natural language overview."""
        data_types = self.detect_data_types(df)
        all_categorical = data_types['categorical_low'] + data_types['categorical_high']
        natural_language_text = self.generate_natural_language_summary(df, data_types)

        summary = [
            html.H4("Dataset Overview"),
            dbc.Card(dbc.CardBody([
                html.P(natural_language_text, className="fst-italic"),
                html.Hr(),
                html.P([
                    f"This dataset contains {df.shape[0]:,} records with {df.shape[1]} features. "
                ]),
                html.H5("Data Composition", className="mt-4"),
                html.Ul([
                    html.Li(f"Numeric columns: {', '.join(data_types['numeric'])}") if data_types['numeric'] else None,
                    html.Li(f"Categorical columns: {', '.join(all_categorical)}") if all_categorical else None,
                    html.Li(f"DateTime columns: {', '.join(data_types['datetime'])}") if data_types['datetime'] else None,
                    html.Li(f"Boolean columns: {', '.join(data_types['boolean'])}") if data_types['boolean'] else None,
                    html.Li(f"Text/ID columns: {', '.join(data_types['text'])}") if data_types['text'] else None,
                ]),
                html.H5("Data Quality Assessment", className="mt-4"),
                self.generate_quality_summary(df),
                html.H5("Statistical Summary", className="mt-4"),
                self.generate_statistical_summary(df, data_types)
            ]))
        ]
        return summary

    def detect_data_types(self, df):
        """Detect and categorize column types with improved accuracy."""
        data_types = {
            'numeric': [],
            'categorical_low': [],
            'categorical_high': [],
            'datetime': [],
            'boolean': [],
            'text': [],
            'unclassified': []
        }
        for column in df.columns:
            dtype = df[column].dtype
            nunique = df[column].nunique()
            if 'date' in column.lower() or 'time' in column.lower():
                try:
                    pd.to_datetime(df[column], errors='raise')
                    data_types['datetime'].append(column)
                    continue
                except (ValueError, TypeError):
                    pass
            if pd.api.types.is_numeric_dtype(dtype):
                if nunique == 2:
                    data_types['boolean'].append(column)
                else:
                    data_types['numeric'].append(column)
                continue
            if dtype == 'object':
                if nunique == 2:
                    data_types['boolean'].append(column)
                    continue
                unique_ratio = nunique / len(df)
                if unique_ratio < 0.5 and nunique < 50:
                    if nunique <= 15:
                        data_types['categorical_low'].append(column)
                    else:
                        data_types['categorical_high'].append(column)
                else:
                    data_types['text'].append(column)
                continue
            data_types['unclassified'].append(column)
        return data_types

    def setup_layout(self):
        self.app.layout = dbc.Container([
            html.H1("Smart Data Analysis Dashboard", className="text-center my-4"),
            dbc.Card([
                dbc.CardBody([
                    html.H4("Upload Your Data", className="text-center my-4"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'},
                        multiple=False
                    )
                ])
            ], className="mb-4"),
            dbc.Tabs([
                dbc.Tab([dbc.Spinner(html.Div(id='output-data-summary'))], label="Dataset Summary"),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Visualization Controls", className="mt-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.Label("Select Visualization Type:"),
                                    dcc.Dropdown(id='viz-type-dropdown', options=[
                                        {'label': 'Scatter Plot', 'value': 'scatter'},
                                        {'label': 'Line Plot', 'value': 'line'},
                                        {'label': 'Bar Chart', 'value': 'bar'},
                                        {'label': 'Box Plot', 'value': 'box'},
                                        {'label': 'Histogram', 'value': 'histogram'},
                                        {'label': 'Heatmap', 'value': 'heatmap'}
                                    ], value='scatter'),
                                    html.Label("X-Axis:", className="mt-3"),
                                    dcc.Dropdown(id='x-axis-dropdown'),
                                    html.Label("Y-Axis:", className="mt-3"),
                                    dcc.Dropdown(id='y-axis-dropdown'),
                                    html.Label("Color By (Optional):", className="mt-3"),
                                    dcc.Dropdown(id='color-dropdown', clearable=True)
                                ])
                            ])
                        ], width=3),
                        dbc.Col([dbc.Spinner(html.Div(id='visualization-output'))], width=9)
                    ])
                ], label="Interactive Visualizations"),
                dbc.Tab([dbc.Spinner(html.Div(id='output-suggested-viz'))], label="Suggested Visualizations"),
                dbc.Tab([dbc.Spinner(html.Div(id='history-content'))], label="History")
            ], className="mb-4"),
            html.Div(id='error-display')
        ])

    def parse_upload(self, contents, filename):
        """Parse uploaded file contents"""
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename.lower():
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename.lower():
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
            return df
        except Exception as e:
            print(f"Error parsing file: {str(e)}")
            return None

    def generate_quality_summary(self, df):
        """Generate data quality information, including duplicates and zero variance."""
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        duplicate_rows = df.duplicated().sum()
        zero_variance_cols = [col for col in df.columns if df[col].nunique() == 1]
        quality_items = [
            html.P(f"Completeness: {(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}% of all data points are present."),
            html.P(f"Uniqueness: Found {duplicate_rows} duplicate rows ({ (duplicate_rows/len(df))*100:.2f}% of the dataset).") if duplicate_rows > 0 else html.P("Uniqueness: No duplicate rows found."),
        ]
        if not missing_cols.empty:
            quality_items.append(html.P("Columns with missing values:"))
            quality_items.append(html.Ul([html.Li(f"{col}: {count} missing values ({(count/df.shape[0])*100:.1f}%)") for col, count in missing_cols.items()]))
        if zero_variance_cols:
            quality_items.append(html.P("Columns with no variance (only one unique value):"))
            quality_items.append(html.Ul([html.Li(col) for col in zero_variance_cols]))
        return html.Div(quality_items)

    def generate_statistical_summary(self, df, data_types):
        """Generate statistical summary for different data types."""
        summaries = []
        all_categorical = data_types['categorical_low'] + data_types['categorical_high']
        if data_types['numeric']:
            numeric_summary = df[data_types['numeric']].describe().T
            numeric_summary['skew'] = df[data_types['numeric']].skew()
            numeric_summary['kurtosis'] = df[data_types['numeric']].kurtosis()
            summaries.append(html.Div([
                html.P("Numeric Columns Summary:"),
                dbc.Table.from_dataframe(numeric_summary.round(2), striped=True, bordered=True, hover=True, responsive=True, className="mb-3")
            ]))
        if all_categorical:
            cat_summary = []
            for col in all_categorical:
                cat_summary.append({
                    'Column': col, 'Unique Values': df[col].nunique(),
                    'Top Value': df[col].mode()[0] if not df[col].mode().empty else 'N/A',
                    'Frequency': df[col].value_counts().max() if not df[col].value_counts().empty else 0
                })
            cat_df = pd.DataFrame(cat_summary)
            summaries.append(html.Div([
                html.P("Categorical Columns Summary:"),
                dbc.Table.from_dataframe(cat_df, striped=True, bordered=True, hover=True, responsive=True, className="mb-3")
            ]))
        if data_types['datetime']:
            dt_summary = []
            for col in data_types['datetime']:
                dt_series = pd.to_datetime(df[col])
                dt_summary.append({
                    'Column': col, 'Start Date': dt_series.min().strftime('%Y-%m-%d'),
                    'End Date': dt_series.max().strftime('%Y-%m-%d'),
                    'Duration (Days)': (dt_series.max() - dt_series.min()).days
                })
            dt_df = pd.DataFrame(dt_summary)
            summaries.append(html.Div([
                html.P("DateTime Columns Summary:"),
                dbc.Table.from_dataframe(dt_df, striped=True, bordered=True, hover=True, responsive=True, className="mb-3")
            ]))
        return html.Div(summaries)

    def create_visualization(self, df, viz_type, x_col, y_col, color_col=None):
        """Create visualization based on selected parameters"""
        try:
            if viz_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot: {y_col} vs {x_col}")
            elif viz_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"Line Plot: {y_col} vs {x_col}")
            elif viz_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"Bar Chart: {y_col} by {x_col}")
            elif viz_type == 'box':
                fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Box Plot: {y_col} by {x_col}")
            elif viz_type == 'histogram':
                fig = px.histogram(df, x=x_col, color=color_col, title=f"Histogram of {x_col}")
            elif viz_type == 'heatmap':
                numeric_df = df.select_dtypes(include=np.number)
                if len(numeric_df.columns) < 2:
                    return html.Div("A heatmap requires at least two numeric columns.")
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap of Numeric Columns")
            fig.update_layout(template='plotly_white', title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))
            return dcc.Graph(figure=fig)
        except Exception as e:
            return html.Div(f"Error creating visualization: {str(e)}")

    def suggest_visualizations(self, df):
        """Suggest the most appropriate visualizations based on data analysis."""
        data_types = self.detect_data_types(df)
        all_categorical = data_types['categorical_low'] + data_types['categorical_high']
        recommendations = []
        if data_types['datetime'] and data_types['numeric']:
            datetime_col = data_types['datetime'][0]
            for numeric_col in data_types['numeric'][:2]:
                try:
                    fig = px.line(df, x=datetime_col, y=numeric_col, title=f"Time Series: {numeric_col} over time")
                    recommendations.append({'score': 1.0, 'component': dbc.Card(dbc.CardBody([html.H5("Time Series Analysis"), html.P(f"Shows how {numeric_col} changes over time."), dcc.Graph(figure=fig)]), className="mb-4")})
                except Exception as e: print(f"Error suggesting Time Series: {e}")
        if data_types['numeric'] and data_types['categorical_low']:
            numeric_col = data_types['numeric'][0]
            for cat_col in data_types['categorical_low'][:2]:
                try:
                    fig = px.box(df, x=cat_col, y=numeric_col, title=f"Distribution of {numeric_col} by {cat_col}")
                    recommendations.append({'score': 0.9, 'component': dbc.Card(dbc.CardBody([html.H5("Numeric vs. Categorical"), html.P(f"Compares the distribution of '{numeric_col}' across different '{cat_col}' categories."), dcc.Graph(figure=fig)]), className="mb-4")})
                except Exception as e: print(f"Error suggesting Box Plot: {e}")
        if len(data_types['categorical_low']) >= 2:
            cat1, cat2 = data_types['categorical_low'][0], data_types['categorical_low'][1]
            try:
                fig = px.bar(df, x=cat1, color=cat2, title=f"Relationship between {cat1} and {cat2}", barmode='group')
                recommendations.append({'score': 0.85, 'component': dbc.Card(dbc.CardBody([html.H5("Categorical Relationship"), html.P(f"Shows the count of '{cat1}' grouped by '{cat2}'."), dcc.Graph(figure=fig)]), className="mb-4")})
            except Exception as e: print(f"Error suggesting Grouped Bar: {e}")
        if len(data_types['numeric']) > 1:
            try:
                numeric_cols = data_types['numeric'][:10]
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap")
                recommendations.append({'score': 0.8, 'component': dbc.Card(dbc.CardBody([html.H5("Correlation Heatmap"), html.P("Shows the correlation between numeric columns. Values near 1 or -1 indicate a strong relationship."), dcc.Graph(figure=fig)]), className="mb-4")})
            except Exception as e: print(f"Error suggesting Heatmap: {e}")
            try:
                corr_matrix = df[data_types['numeric']].corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                max_corr_pair = corr_matrix.stack().idxmax()
                var1, var2 = max_corr_pair
                fig = px.scatter(df, x=var1, y=var2, trendline="ols", title=f"Strongest Correlation: {var1} vs {var2}")
                recommendations.append({'score': 0.95, 'component': dbc.Card(dbc.CardBody([html.H5("Strongest Numeric Relationship"), html.P(f"A strong linear relationship was found between {var1} and {var2}."), dcc.Graph(figure=fig)]), className="mb-4")})
            except Exception as e: print(f"Error suggesting Correlation Scatter: {e}")
        if data_types['numeric']:
            try:
                col = data_types['numeric'][0]
                fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
                recommendations.append({'score': 0.7, 'component': dbc.Card(dbc.CardBody([html.H5(f"Numeric Distribution"), html.P(f"Shows the frequency distribution of {col}."), dcc.Graph(figure=fig)]), className="mb-4")})
            except Exception as e: print(f"Error suggesting Histogram: {e}")
        if data_types['categorical_low']:
            try:
                col = data_types['categorical_low'][0]
                counts_df = df[col].value_counts().reset_index()
                x_col_name = counts_df.columns[0]
                y_col_name = counts_df.columns[1]
                fig = px.bar(counts_df, x=x_col_name, y=y_col_name, title=f"Counts for {col}")
                fig.update_xaxes(title_text=col).update_yaxes(title_text='Count')
                recommendations.append({'score': 0.65, 'component': dbc.Card(dbc.CardBody([html.H5(f"Categorical Counts"), html.P(f"Shows the number of occurrences for each category in {col}."), dcc.Graph(figure=fig)]), className="mb-4")})
            except Exception as e: print(f"Error suggesting Bar Chart: {e}")
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        suggestions_layout = [html.H4("Recommended Visualizations", className="mb-4")]
        if recommendations:
            suggestions_layout.extend([rec['component'] for rec in recommendations])
        else:
            suggestions_layout.append(dbc.Alert("No specific patterns were automatically detected. Try exploring the data manually.", color="warning"))
        return html.Div(suggestions_layout)

    def setup_callbacks(self):
        @self.app.callback(
            [Output('x-axis-dropdown', 'options'), Output('y-axis-dropdown', 'options'), Output('color-dropdown', 'options'), Output('x-axis-dropdown', 'value'), Output('y-axis-dropdown', 'value')],
            Input('upload-data', 'contents'),
            State('upload-data', 'filename')
        )
        def update_dropdowns(contents, filename):
            if contents is None:
                return [], [], [], None, None
            try:
                df = self.parse_upload(contents, filename)
                if df is None: raise ValueError("Failed to parse file")
                self.df = df
                self.filename = filename
                options = [{'label': col, 'value': col} for col in df.columns]
                default_x = df.columns[0] if len(df.columns) > 0 else None
                default_y = df.columns[1] if len(df.columns) > 1 else default_x
                return options, options, options, default_x, default_y
            except Exception as e:
                print(f"Error updating dropdowns: {str(e)}")
                return [], [], [], None, None

        @self.app.callback(
            [Output('output-data-summary', 'children'), Output('output-suggested-viz', 'children')],
            Input('upload-data', 'contents'),
            State('upload-data', 'filename')
        )
        def update_output(contents, filename):
            if contents is None:
                return html.Div("Please upload a CSV or Excel file to begin.", className="text-center p-4"), None
            try:
                df = self.parse_upload(contents, filename)
                if df is None: raise ValueError("Failed to parse file")
                self.df = df
                self.filename = filename
                summary = self.generate_dataset_summary(df)
                suggestions = self.suggest_visualizations(df)
                try:
                    if "user" in session:
                        user_email = session["user"]
                        summary_text = f"Uploaded '{filename}': {df.shape[0]} rows, {df.shape[1]} columns."
                        history_data = {"user_email": user_email, "filename": filename, "summary": summary_text, "timestamp": firestore.SERVER_TIMESTAMP, "action": "upload"}
                        db.collection("history").add(history_data)
                except Exception as firestore_error:
                    print(f"Error saving upload to Firestore: {str(firestore_error)}")
                return summary, suggestions
            except Exception as e:
                error_msg = dbc.Alert(f"Error processing file: {str(e)}", color="danger", dismissable=True)
                return error_msg, error_msg

        @self.app.callback(Output('history-content', 'children'), Input('upload-data', 'contents'))
        def update_history_tab(contents):
            if "user" not in session:
                return html.Div("Please log in to view your history.", className="text-danger")
            try:
                user_email = session["user"]
                history_ref = db.collection("history").where("user_email", "==", user_email).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20)
                history_docs = history_ref.stream()
                history_items = []
                for doc in history_docs:
                    history = doc.to_dict()
                    timestamp = history.get('timestamp')
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(timestamp, 'strftime') else "Just now"
                    history_items.append(dbc.ListGroupItem([html.Div([html.Strong(f"{history.get('summary', 'N/A')}"), html.Small(timestamp_str, className="text-muted, float-end")])]))
                if not history_items:
                    return html.Div("No history found. Upload some data to get started!", className="text-muted p-3")
                return dbc.ListGroup(history_items)
            except Exception as firestore_error:
                return html.Div(f"Error retrieving history: {str(firestore_error)}", className="text-danger")

        @self.app.callback(
            Output('visualization-output', 'children'),
            [Input('viz-type-dropdown', 'value'), Input('x-axis-dropdown', 'value'), Input('y-axis-dropdown', 'value'), Input('color-dropdown', 'value')],
        )
        def update_visualization(viz_type, x_col, y_col, color_col):
            if self.df is None: return html.Div("Please upload data first", className="text-warning p-3, text-center")
            if not viz_type or not x_col: return html.Div("Please select visualization parameters", className="text-warning p-3, text-center")
            try:
                visualization_component = self.create_visualization(self.df, viz_type, x_col, y_col, color_col)
                try:
                    if "user" in session:
                        user_email = session["user"]
                        summary = f"Created '{viz_type}' plot"
                        if y_col: summary += f" of {y_col} vs {x_col}"
                        else: summary += f" of {x_col}"
                        if color_col: summary += f", colored by {color_col}"
                        history_data = {"user_email": user_email, "filename": self.filename if self.filename else "N/A", "summary": summary, "timestamp": firestore.SERVER_TIMESTAMP, "action": "visualization"}
                        db.collection("history").add(history_data)
                except Exception as firestore_error:
                    print(f"Error saving to Firestore: {str(firestore_error)}")
                return visualization_component
            except Exception as e:
                return html.Div(f"Error creating visualization: {str(e)}", className="text-danger")

# Flask Routes for Login/Signup
@flask_app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@flask_app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth_client.create_user_with_email_and_password(email, password)
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("home"))
        except Exception as e:
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message:
                error_message = "This email is already in use. Please log in or use a different email."
            else:
                error_message = "An error occurred during signup. Please try again."
            return render_template("signup.html", error=error_message)
    return render_template("signup.html", error=None)

@flask_app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth_client.sign_in_with_email_and_password(email, password)
            session["user"] = email
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            flash(f"Login failed: Invalid email or password", "danger")
    return render_template("login.html")

@flask_app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@flask_app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for("home"))
    return render_template("dashboard.html")

@flask_app.route("/history")
def history():
    if "user" not in session:
        flash("Please log in to view your history.", "warning")
        return redirect(url_for("home"))
    try:
        user_email = session["user"]
        history_ref = db.collection("history").where("user_email", "==", user_email).order_by("timestamp", direction=firestore.Query.DESCENDING)
        history_docs = history_ref.stream()
        history_data = []
        for doc in history_docs:
            history_dict = doc.to_dict()
            timestamp = history_dict.get('timestamp')
            if hasattr(timestamp, 'strftime'):
                history_dict['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            history_data.append(history_dict)
        return render_template("history.html", history=history_data)
    except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return render_template("history.html", history=[])

# Run the app
if __name__ == "__main__":
    dashboard = Dashboard()
    flask_app.run(debug=True, port=5001)