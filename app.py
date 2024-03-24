import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
from plotly.express import box
import plotly.graph_objs as go
from plotly.subplots import make_subplots

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

finance = pd.concat([train, test])

# Split the values at the first comma and take the first part
finance['Type_of_Loan'] = finance['Type_of_Loan'].str.split(',', n=1).str[0]

# Top features
top_features = ['Credit_Card',
                'Credit_History_Age',
                'Num_of_Loan',
                'Changed_Credit_Limit',
                'Payment_of_Min_Amount_No',
                'Delay_from_due_date',
                'Payment_of_Min_Amount_Yes',
                'Outstanding_Debt',
                'Interest_Rate',
                'Delayed_Payment']

# Corresponding importances
top_importances = [0.0396712001699896,
                   0.043669210501149074,
                   0.054657125388645714,
                   0.05721376551970896,
                   0.07072898048424679,
                   0.09007179254032731,
                   0.09101891584090184,
                   0.09626828003921105,
                   0.13212492259246894,
                   0.13971806579283513]

# Create DataFrame
top_features_df = pd.DataFrame({'Feature': top_features, 'Importance': top_importances})

# Create the Plotly figure for feature importance
fig = go.Figure(go.Bar(
    x=top_features_df['Importance'],
    y=top_features_df['Feature'],
    orientation='h'
))

# Update layout
fig.update_layout(
    title='Top 10 Feature Importances',
    xaxis_title='Feature Importance',
    yaxis_title='Feature',
    yaxis=dict(autorange="reversed")  # Reverse y-axis to display the most important feature at the top
)

# Selecting specific numerical columns
selected_columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card', 'Num_of_Loan', 'Interest_Rate', 'Num_of_Delayed_Payment', 'Outstanding_Debt']
numerical_columns = finance[selected_columns]

summary_stats = {
    "": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    "Age": [150000.000000, 33.479060, 10.767158, 14.000000, 25.000000, 33.000000, 42.000000, 56.000000],
    "Annual Income": [150000.000000, 50505.123449, 38299.358260, 7005.930000, 19342.972500, 36999.705000, 71683.470000, 179987.280000],
    "Monthly Salary": [150000.000000, 4192.525296, 3182.728812, 303.645417, 1625.558333, 3091.000000, 5951.373333, 15204.633333],
    "No of CreditCards": [150000.000000, 5.533660, 2.068672, 0.000000, 4.000000, 5.000000, 7.000000, 11.000000],
    "Interest Rate": [150000.000000, 14.532080, 8.741316, 1.000000, 7.000000, 13.000000, 20.000000, 34.000000],
    "No of Loan": [150000.000000, 3.532880, 2.446352, 0.000000, 2.000000, 3.000000, 5.000000, 9.000000],
    "Delay from due date": [150000.000000, 21.076753, 14.804854, 0.000000, 10.000000, 18.000000, 28.000000, 62.000000],
    "Monthtly Investment": [150000.000000, 55.101315, 39.006867, 0.000000, 27.959111, 45.156550, 71.295797, 434.191089],
    "Monthly Balance": [150000.000000, 392.937052, 201.716752, 0.007760, 267.720142, 333.921243, 463.423068, 1183.930696]
}

numerical_columns = finance.select_dtypes(include=['float64', 'int64'])
columns_to_drop = ['ID', 'Customer_ID', 'SSN','Month','Credit_Utilization_Ratio']
numerical_columns = numerical_columns.drop(columns_to_drop, axis=1)

# Exclude non-numeric columns before computing the correlation matrix
numeric_finance = finance.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numeric_finance.corr()

# Get the column names from the correlation matrix
column_names = correlation_matrix.columns

heatmap = go.Heatmap(
    z=correlation_matrix.values,
    x=column_names,
    y=column_names,
    colorscale='RdBu',
    zmin=-1,
    zmax=1,
    colorbar=dict(title='Correlation', tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=['-1', '-0.5', '0', '0.5', '1']),
    hoverinfo='z', 
)

annotations = []
for i, row in enumerate(correlation_matrix.values):
    for j, value in enumerate(row):
        annotations.append(dict(
            text=str(round(value, 2)),  
            x=column_names[j], 
            y=column_names[i], 
            xref='x',
            yref='y',
            showarrow=False,
            font=dict(color='black'), 
        ))

layout = go.Layout(
    title='Correlation Matrix',
    xaxis=dict(tickvals=list(range(len(column_names))), ticktext=column_names, tickangle=45, range=[-0.5, len(column_names)-0.5]),
    yaxis=dict(tickvals=list(range(len(column_names))), ticktext=column_names, range=[len(column_names)-0.5, -0.5]),
    height=1000,  
    width=1000,  
    annotations=annotations, 
)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    
    html.Div([
        html.H1("Finance Analysis Dashboard"),
        html.P("This project, I conducted analysis of financial data, employing statistical techniques and machine learning algorithms to gain insights and develop predictive models. Exploratory data analysis, I identified key variables and relationships, providing crucial context for modeling . Utilizing descriptive statistics and visualization techniques.I built predictive models using regression analysis, time series forecasting, and machine learning algorithms using random forests. Through iterative testing, validation, and refinement, I optimized these models for accuracy and reliability, ensuring they provide actionable insights for strategic decision-making. This project represents a culmination of rigorous data analysis and modeling, aimed at empowering stakeholders with actionable insights for financial management."),
        html.Div([
        html.Button('Summary', id='btn-4', n_clicks=0),
        html.Button('Correlation', id='btn-1', n_clicks=0),
        html.Button('Visualization', id='btn-2', n_clicks=0),
        html.Button('Model', id='btn-3', n_clicks=0)
                ], className="buttons"),

        html.Div([
            html.A(
            html.Img(src="/assets/website.png", className="icon", id="website-icon"),
            href="https://yourwebsite.com",
            target="_blank",
            style={'display': 'inline-block', 'position': 'absolute', 'left': 'calc(100% - 80px)'}  
            ),
            html.A(
            html.Img(src="/assets/twitter.png", className="icon", id="twitter-icon"),
            href="https://yourwebsite.com",
            target="_blank",
            style={'display': 'inline-block', 'position': 'absolute', 'left': 'calc(100% - 55px)'}  
            )
        ], className="icons")
    ], className="header"),
    
    html.Div(className="divider"),
    
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(dash.dependencies.Output('url', 'pathname'),
              [dash.dependencies.Input('btn-1', 'n_clicks'),
               dash.dependencies.Input('btn-2', 'n_clicks'),
               dash.dependencies.Input('btn-3', 'n_clicks'),
               dash.dependencies.Input('btn-4', 'n_clicks')])
def update_pathname(btn1_clicks, btn2_clicks, btn3_clicks, btn4_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return '/'
    else:
        btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if btn_id == 'btn-1':
            return '/page-1'
        elif btn_id == 'btn-2':
            return '/page-2'
        elif btn_id == 'btn-3':
            return '/page-3'
        elif btn_id == 'btn-4':
            return '/page-4'
        else:
            return '/'

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        # Generate heatmap
        heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=column_names,
            y=column_names,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation', tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=['-1', '-0.5', '0', '0.5', '1']),
            hoverinfo='z', 
        ))

        # Add annotations
        annotations = []
        for i, row in enumerate(correlation_matrix.values):
            for j, value in enumerate(row):
                annotations.append(dict(
                    text=str(round(value, 2)),  
                    x=column_names[j], 
                    y=column_names[i], 
                    xref='x',
                    yref='y',
                    showarrow=False,
                    font=dict(color='black')
                ))

        # Update layout
        heatmap.update_layout(
            title='Correlation Matrix',
            xaxis=dict(tickvals=list(range(len(column_names))), ticktext=column_names, tickangle=45, range=[-0.5, len(column_names)-0.5]),
            yaxis=dict(tickvals=list(range(len(column_names))), ticktext=column_names, range=[len(column_names)-0.5, -0.5]),
            height=1000,  
            width=1000,  
            annotations=annotations, 
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        return html.Div([
            html.H2('Page 1'),
            dcc.Graph(figure=heatmap)
        ])

    elif pathname == '/page-2':
        # Create a figure with 3 rows and 2 columns
        fig = make_subplots(rows=4, cols=2, subplot_titles=('Age', 'Annual Income', 'Monthly Inhand Salary', 'Num Credit Card', 'Num_of_Loan', 'Interest Rate', 'Num of Delayed Payment', 'Outstanding Debt'), 
                        vertical_spacing=0.1, horizontal_spacing=0.1, 
                        shared_xaxes=False, shared_yaxes=False, 
                        column_widths=[0.5, 0.5], row_heights=[0.5, 0.5, 0.5, 0.5], 
                        specs=[[{"type": "histogram"}, {"type": "histogram"}], 
                            [{"type": "histogram"}, {"type": "histogram"}],
                            [{"type": "histogram"}, {"type": "histogram"}],
                            [{"type": "histogram"}, {"type": "histogram"}]])

        # Iterate through selected columns and create plots, assigning them to the appropriate subplot
        for i, column in enumerate(selected_columns, start=1):
            row = (i - 1) // 2 + 1  
            col = (i - 1) % 2 + 1  
            fig.add_trace(go.Histogram(x=numerical_columns[column], name=column, nbinsx=30), row=row, col=col)

        # Update layout
        fig.update_layout(title_text="Distribution of Numerical Columns", showlegend=False, width=1000, height=1200, plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',)

        # Update x-axis titles for all subplots
        for i in range(1, 5):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Value", row=i, col=j)

        return html.Div([
            html.H2('Distribution by some Variables'),
            dcc.Graph(figure=fig)
        ])
    elif pathname == '/page-3':

        # Create bar plot
        feature_importance_fig = go.Figure(go.Bar(
            x=top_features_df['Importance'],
            y=top_features_df['Feature'],
            orientation='h'
        ))
        
        # Update layout
        feature_importance_fig.update_layout(
            title='Top 10 Feature Importances',
            xaxis_title='Feature Importance',
            yaxis_title='Feature',
            margin=dict(l=100, r=20, t=70, b=70),  # Adjust margins as needed
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width = 700,
            height =500
        )

        return html.Div([
            html.H2('Case Study Results:', style={'font-family': 'cursive'}),
            html.H2('Creating State-of-the-Art Models', style={'font-family': 'cursive'}),
            # Add your Dash components for Page 3 here
            html.P('Through a systematic approach, we successfully built and evaluated a Random Forest classifier to predict credit score, utilizing a dataset containing various financial features.The study gave us positive results. We were able to find the most important variables for credit assessment'),
            
            # Add the confusion matrix plot here
            dcc.Graph(
                id='confusion-matrix',
                figure={
                    'data': [{
                        'type': 'heatmap',
                        'z': [[7195, 0, 17], [0, 9093, 54], [31, 38, 13572]],  # Update with your confusion matrix values
                        'x': ['Standard', 'Good', 'Bad'],
                        'y': ['Standard', 'Good', 'Bad'],
                        'colorscale': 'Blues',
                        'showscale': False  # Remove color legend
                    }],
                    'layout': {
                        'title': 'Confusion Matrix',
                        'xaxis': {'title': 'Predicted Labels'},
                        'yaxis': {'title': 'True Labels'},
                        'annotations': [
                            {
                                'x': j,
                                'y': i,
                                'text': str(val),
                                'showarrow': False,
                                'font': {'color': 'white'} 
                            } for i, row in enumerate([[7195, 0, 17], [0, 9093, 54], [31, 38, 13572]]) for j, val in enumerate(row)
                        ],
                        'paper_bgcolor': 'rgba(0,0,0,0)',  
                        'plot_bgcolor': 'rgba(0,0,0,0)' ,
                        'width': 450,
                        'height': 500
                    }
                }
            ),
            html.P('Further refinement was achieved through hyperparameter tuning, which optimized the model performance. Additionally, feature importance analysis provided valuable insights into the significant factors influencing credit mix predictions. Overall, our approach demonstrated a robust methodology for credit mix prediction, incorporating data preprocessing, model training, evaluation, and refinement, culminating in a reliable and interpretable predictive model.'),
            
            # Add the feature importance plot here
            dcc.Graph(
                id='feature-importance',
                figure=feature_importance_fig
            )
        ])
    elif pathname == '/page-4':

        return html.Div([
        html.H2('Summary:'),
        html.P('The provided summary presents key descriptive statistics for various financial attributes within the dataset. Each attribute, such as "Age," "Annual Income," and "Interest Rate," is accompanied by statistics including count, mean, standard deviation, minimum, quartiles, and maximum values. For instance, the average age of individuals in the dataset is approximately 33 years, with a standard deviation of around 10.77 years. Similarly, the annual income exhibits a mean of approximately $50,505, with a standard deviation of $38,299. The data also reveals insights into financial behaviors, such as the average number of credit cards held by individuals, the prevailing interest rates, and the typical delay from due dates on financial obligations. Moreover, details on monthly investments, loan counts, and monthly balances provide further granularity regarding financial activities within the dataset. This summary serves as a comprehensive overview, offering valuable insights into the financial landscape captured by the dataset.'),
        dash_table.DataTable(
            id='summary-table',
            columns=[{'name': col, 'id': col} for col in summary_stats.keys()],
            data=[{col: summary_stats[col][i] for col in summary_stats.keys()} for i in range(len(summary_stats[list(summary_stats.keys())[0]]))],
            style_table={'backgroundColor': 'rgba(0,0,0,0)'},  
            style_cell={'backgroundColor': 'rgba(0,0,0,0)', 'color': 'black'} 
        )
    ])

    else:
        # Assuming 'finance' is your pandas DataFrame
        fig2 = box(finance, x="Type_of_Loan", y="Annual_Income", title="Annual_Income by Type of Loan")

        # Customize labels (similar to plt.xlabel and plt.ylabel)
        fig2.update_xaxes(title_text="Type of Loan")
        fig2.update_yaxes(title_text="Annual Income")

        # Rotate x-axis labels for better readability
        fig2.update_layout(xaxis_tickangle=25)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height= 400
                            )
        
                # Assuming 'finance' is your pandas DataFrame
        fig3 = box(finance, x="Occupation", y="Annual_Income", title="Distribution of Annual Income by Occupation")

        # Customize labels (similar to plt.xlabel and plt.ylabel)
        fig3.update_xaxes(title_text="Occupation")
        fig3.update_yaxes(title_text="Annual Income")

        # Rotate x-axis labels for better readability
        fig3.update_layout(xaxis_tickangle=25)
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height= 400
                            )

        # Embed Plotly graph within children of the Div
        return html.Div(
        style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between', 'width': '100%', 'padding': '1rem'},  # Change flex-direction to column
        children=[
            html.Div(
                style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'width': '100%'},  # Inline styling for flexbox layout and width
                children=[
                    html.Div(
                        style={'width': '40%'},  
                        children=[
                            html.H2('Machine-Learning Based credit score prediction', style={'font-family': 'cursive'}),
                            html.H4('Visualization'),
                            html.P('In our exploration of the financial data, we leveraged visualization techniques to gain a deeper understanding of the relationships between the 25 variables using histograms, heatmaps, scatter plots, and other types of plots. A breakdown of our approach is in correlation and Visualization here is an example of:'),
                        ]
                    ),
                    dcc.Graph(id='example-plot', figure=fig2, style={'width': '50%'})  
                ]
            ),
            html.Div(
                style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'width': '100%'},  # Inline styling for flexbox layout and width
                children=[
                    html.Div(
                        style={'width': '55%'},  
                        children=[
                            html.P('Another Example of plot to help visualize Income by Occupation'),
                            dcc.Graph(id='example-plot', figure=fig3) 
                        ]
                    ),
                    html.Div(
                        style={'width': '40%'},  
                        children=[
                            html.P([
                            'What problems did we solve that can be applied to other projects?',
                            html.Br(), 
                            '- Extensive analysis of predictive models',
                            html.Br(),  
                            '- Addressing classification challenges with limited data in credit scoring',
                            html.Br(),
                            '- Solving classification a problem with Credit assesment (Good, Bad, Standard)',
                            html.Br(),
                            '- Conducting extensive analysis for credit risk assessment models',
                            html.Br(),
                            '- Processing and summarizing large volumes of financial data for credit assessment purposes',
                            html.Br(),
                            '- Processing of large amounts of complex text - summarization',
                            html.Br(),
                            '- Enhancing explainability of credit scoring model decisions for stakeholders',
                        ])

                        ]
                    )
                ]
            )
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=True)




