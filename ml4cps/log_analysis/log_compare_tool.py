# Author Nemanja Hranisavljevic
# Work within the project SILK:
#

import json
import spacy
import dash
from dash import dcc, html, dash_table, Output, Input, State
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from plotly import graph_objs as go
import pandas as pd
import base64
import numpy as np


# Initialize the Dash app
app = dash.Dash(__name__, title="Log Compare Tool")

app.layout = html.Div(id="root",
    children = [html.Div(id="content", children=[
        html.Div(
            children=[ 
                html.Img(src='/assets/helmut.png', style={'max-height': '80px',
                                                          'width': 'auto',
                                                          'margin-right': '30px',
                                                          'margin-left': '15px',
                                                          'margin-top': '15px'}),
                html.Img(src='/assets/aicps.jpg', style={'max-height': '80px',
                                                         'width': 'auto',
                                                         'margin-top': '15px'}),
            ]
        ),
        html.H1("Event Extraction Comparison", 
                style = {
                    'text-align': 'center'
                }),
        dcc.Store(id='stored-json-data', data={'ground_truth': None,
                                               'others': {}}),
        dcc.Upload(
            id='upload-gt',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '99%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '99%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ), 
        html.H2("Cosine similarity graph",
                style={
                    'text-align': 'center'
                }),
        dcc.Graph(id='cosplot', figure={
            'layout': go.Layout(
                    plot_bgcolor='#e5ecf6',
                    paper_bgcolor='#f9f9f9',  
                    font={'color': '#333333'}, 
                    title='Styled Graph',
                    xaxis={'title': 'X Axis', 'showgrid': False, 'color': 'black'}, 
                    yaxis={'title': 'Y Axis', 'showgrid': False, 'color': 'black'},  
                    margin={'l': 40, 'b': 40, 't': 40, 'r': 0},  
                )
        }, style={
            'padding': '30px',
            # 'width': '100%',
            'height': '400px',
            }),
        html.Button(
            id='edit-button',
            children = "Download ground truth file:",
            style = {
                'width': '97vw',
                'height': '60px',
                'font-size': '16px'
            }
        ),
        dcc.Download(id='download-json'),
        dash_table.DataTable(
            id='datatable',
            columns=[],
            data=[],
            editable=True,
            style_table={
                'padding': '30px',
                'max-width': '95%',
                'height': 'auto',
                'overflowX': 'auto',
                'overflowY': 'hidden',
                # 'margin-left': '30px',
                # 'margin-right': '30px',
                'color': 'black'
            },
            style_data={
                'whiteSpace': 'normal', 
                'height': 'auto',
                'overflow': 'hidden'
            },
            style_cell={
                'fontSize': '12px',
                'maxWidth': 0  
            }
        )]),
        html.Div(id="footer",
            children = 'Project SILK 2024')
    ]

)
nlp = spacy.load('en_core_web_md')

# Based on the cosine similarity score, 0.20 can be kept as a threshold
# Values >0.20 match the strings good.
def string_similarity(str1, str2):
    # Tokenize and preprocess the strings
    doc1 = nlp(str(str1))
    doc2 = nlp(str(str2))
    # Use TF-IDF to vectorize the preprocessed strings
    str1 = np.str_(str1)
    str2 = np.str_(str2)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2])
    # Calculate cosine similarity between the vectors
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim

def parse_json(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Load JSON file into a DataFrame
        data = json.loads(decoded.decode('utf-8'))
        df = pd.json_normalize(data)
    except Exception as e:
        print(e)
        return None

    return df

def json_to_df(contents, filenames):
    if contents:
        data_dicts = {}
        for content, filename in zip(contents, filenames):
            # Decode the content
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'json' in filename:
                    # Parse the JSON file
                    json_data = json.loads(decoded.decode('utf-8'))
                    # Convert JSON to DataFrame
                    df = pd.DataFrame.from_dict(json_data, orient='index')
                    data_dicts[filename] = df # .to_dict(orient="records")
                else:
                    return None
            except Exception as e:
                print(e)
                return None
        return data_dicts

    return None

#defines colors used in bar chart 
color_arr = ['blue', 'red', 'yellow', 'green', 'purple']

#defines column order for datatable 
columnorder = ["index", "symbol", "template", "sample_message", "significance"]

@app.callback(
    Output('download-json', 'data'),
    Input('edit-button', 'n_clicks'),
    State('datatable', 'data'),
    State('datatable', 'columns'),
    State('upload-gt', 'children')
)
def update_files(n_clicks, data, columns, gt_file):
    if n_clicks != None:
        if columns != []: 
            col_order = ["symbol", "template", "sample_message",  "significance"]
            df = pd.DataFrame(data)
            named_data_dict = {str(row["index"]): row[col_order].to_dict() for _, row in df.iterrows()}
            json_file = json.dumps(named_data_dict, indent = 4)
            return dict(content=json_file, filename=gt_file, mime_type="application/json")
    return ""

@app.callback(
    Output('datatable', 'columns'),
    Output('datatable', 'data'),
    Output('cosplot', 'figure'),
    Input('stored-json-data', 'data'),
    prevent_initial_call=True
)
def update_other(stored_data):
    # We store all content of files in the store, now we use it for visualization
    #columnorder1 = ["index", "symbol", "expected_symbol", "template", "expected_template", "sample_message", "significance"]
    columnorder1 = []
    if stored_data and "ground_truth" in stored_data and stored_data["ground_truth"]:
        gt = stored_data["ground_truth"]
        df_gt = {k: pd.DataFrame.from_dict(v, orient="index") for k, v in gt.items()}
        index_arr = []
        for k, v in gt.items():
            for ind in v:
                index_arr.append(ind)

        if "others" in stored_data and stored_data["others"]:
            others = stored_data["others"]
            df_others = {k: pd.DataFrame.from_dict(v, orient="index") for k, v in others.items()}
        else:
            df_others = {}

        df = create_combined_df(df_gt, df_others)
        df['index'] = index_arr
        fig = {}

        #calculating cosine similarity, rearranging columns in dataframe and creating bar chart
        num_other = len(stored_data['others'])
        if num_other > 0:
            filenames = [filename for filename in stored_data["others"]]
            indexlen = df['index'].count()

            expected_symbolarr = []
            expected_templatearr = []
            symbolorder = []
            templateorder = []
            samplemsgorder = []
            signiforder = []
            similarityorder = []

            #figure initialization
            fig = go.Figure()
            fig.update_layout(
            )
            fig.add_shape(type="line",
                x0=0, x1=1, y0=0.2, y1=0.2,
                line=dict(color="green", width=3),
                xref='paper', yref='y')

            expected_symbol_matrix = []

            for k in range(0, num_other):
                expected_symbolarr = []
                expected_templatearr = []
                for i in range(0, indexlen):
                #creating 'expected symbol' and 'expected template' columns
                #creating 2d array of data for calculating cosine similarity
                    _index = "e" + str(i)
                    if _index in df['index']:
                        msg1 = df.at[_index, f"sample_message_{k + 1}"]
                        added_expected_symbol = False
                        for j in range(0, indexlen):
                            __index = f"e{j}"
                            if i != j:
                                if __index in df['index']:
                                    msg2 = df.at[__index, 'sample_message']
                                    if msg1 == msg2:
                                        expected_symbolarr.append(df.at[__index, 'symbol'])
                                        expected_templatearr.append(df.at[__index, 'template'])
                                        added_expected_symbol = True                          
                        if added_expected_symbol == False:
                            expected_symbolarr.append(None)
                            expected_templatearr.append(None)
                    else:
                        expected_symbolarr.append(None)
                        expected_templatearr.append(None)    
                expected_symbol_matrix.append(expected_symbolarr)  
                if k == 0:
                    df['expected_symbol'] = expected_symbolarr
                    df['expected_template'] = expected_templatearr

            #calculating cosine similarity, inserting the data into a column
            for i in range(0, num_other):
                cosarr = []
                for j in range(0, indexlen):
                    _index = f"e{j}"
                    if _index in df['index']:
                        str1 = expected_symbol_matrix[i][j]
                        str2 = df.at[_index, f"symbol_{i + 1}"]
                        if str1 != None and str2 != None:
                            cosarr.append(string_similarity(str1, str2))
                        else:
                            cosarr.append(0)
                    else:
                        cosarr.append(0)
                df['symbol_similarity_' + str(i + 1)] = cosarr
                cosarr_trace = [0.01 if v == 0 else v for v in cosarr]
                
                #rearranging columns from other uploaded files and updating the figure
                symbolorder.append(f"symbol_{i + 1}")
                templateorder.append(f"template_{i + 1}")
                samplemsgorder.append(f"sample_message_{i + 1}")
                signiforder.append(f"significance_{i + 1}")
                similarityorder.append(f"symbol_similarity_{i + 1}")
                fig.add_trace(go.Bar(
                    x = df['index'],
                    y = cosarr_trace,
                    customdata = cosarr,
                    hovertemplate=(
                        '<b>%{x}</b><br>' + 
                        'Value: %{cosarr}<br>'
                    ),
                    marker_color = color_arr[i],
                    name = filenames[i]
                ))
            columnorder1.append("index")
            columnorder1.append("symbol")
            for sym in symbolorder:
                columnorder1.append(sym)
            columnorder1.append("expected_symbol")
            for sim in similarityorder:
                columnorder1.append(sim)
            columnorder1.append("template")
            for tem in templateorder:
                columnorder1.append(tem)
            columnorder1.append("expected_template")
            columnorder1.append("sample_message")
            for msg in samplemsgorder:
                columnorder1.append(msg)
            columnorder1.append("significance")
            for sig in signiforder:
                columnorder1.append(sig)
            df = df[columnorder1]
        else:
            df = df[columnorder] 
        

        return [{'name': i, 'id': i} for i in df.columns], df.reset_index(drop=False).to_dict(orient="records"), fig
    else:
        return [], [], {}

@app.callback(
    Output('upload-gt', 'children'),
    Output('upload-data', 'children'),
    Output('stored-json-data', 'data'),
    State('upload-gt', 'contents'),
    Input('upload-gt', 'filename'), 
    State('upload-data', 'contents'),
    Input('upload-data', 'filename'),
    State('stored-json-data', 'data'),
    State('upload-gt', 'children'),
    State('upload-data', 'children'),
    prevent_initial_call=False
)
# children_gt, children_data
def update_other(content, filename, contents, filenames, stored_data, children_gt, children_data):
    # Here we store new file content
    children_data = str(children_data)
    if children_data[0] == '{':
        children_data = "Drag and Drop or Select Files"
    if content:
        df = json_to_df([content], [filename])
        children_gt = filename
        if df:
            stored_data['ground_truth'] = {filename: df[filename].to_dict(orient="index")}
    if contents:
        new_content = json_to_df(contents, filenames)
        if new_content:
            if children_data == "Drag and Drop or Select Files":
                children_data = ""
            for k, v in new_content.items():
                stored_data['others'][k] = v.to_dict(orient="index")
            for _filename in filenames:
                if children_data.__contains__(_filename) != True:
                    children_data += str(_filename) + ", "

    return children_gt, children_data, stored_data

def create_combined_df(df_gt, df_others):
    df = pd.concat(df_gt.values(), axis=0)
    for ind, (fn, other) in enumerate(df_others.items()):
        # maybe in future we can use filename (fn)
        df = df.join(other, how='left', rsuffix=(f'_{ind+1}'))
    return df
    # df_others = df_others.rename(columns=lambda x: x + '_2')
    # df_combined = pd.concat([df_gt, df_others], axis=1)
    # columns = df_combined.columns.tolist()
    # original_cols = [col for col in columns if not col.endswith('_2')]
    # renamed_cols = [col for col in columns if col.endswith('_2')]
    # renamed_cols = [col[:-2] for col in renamed_cols]
    # interleaved_cols = []
    # for col in original_cols:
    #     interleaved_cols.append(col)
    #     if col in renamed_cols:
    #         interleaved_cols.append(col + '_2')
    # df_combined = df_combined[interleaved_cols]
    # del df_combined['index_2']
    # return df_combined


def run(port=8050):
    app.run(debug=False, port=port)

if __name__ == '__main__':
    app.run(debug=False)
