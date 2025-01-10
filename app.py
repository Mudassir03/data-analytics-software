from flask import Flask,session ,render_template, request, redirect, url_for
from flask_session import Session
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import io
import os
import seaborn as sb



app = Flask(__name__)



app.config['SESSION_TYPE'] = 'filesystem'  # Or use 'redis', 'mongodb', etc.
app.config['SECRET_KEY'] = "f5fae0566092bb86c9ea9eb8d4e46b8398d23496fcc7d19740c7e7f73d494218"
app.config['SESSION_FILE_DIR'] = './flask_session_files'  # Directory for storing session files
Session(app)


def label_encode_categorical_columns(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df



# Define the route for the file upload page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is uploaded
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            session.pop('uploaded_file', None)
            session.pop('processed_df', None)
            session.pop('fileName', None)
            session.pop('null_count', None)
            session.pop('columnNames', None)
            # Save the file to the upload folder
            file_stream = file.stream.read().decode('utf-8')
            session['uploaded_file']=file_stream
            session['fileName']=file.filename
            
            return render_template('uploaded.html', fileName=file.filename)
        else:
            return render_template('error.html',message="Please upload a valid CSV file.") 

    return render_template('home.html')
@app.route('/dashboard')
def opretions():

    uploaded_file_content = session.get('uploaded_file')
    if not uploaded_file_content:
        return render_template('error.html',message="No uploaded file found in the session!")

    df = pd.read_csv(io.StringIO(uploaded_file_content))
    dfHead=df.head().to_html()
    dfTail=df.tail().to_html()
    buffer = io.StringIO()
    df.info(buf=buffer)
    dfInfo = buffer.getvalue()
    dfSummary=df.describe().to_html()
    nullValue = df.isnull().sum().to_frame(name='Null Count').to_html()
    session['null_count']=nullValue
    #heatMap=sb.heatmap(df.corr(), cmap='heat', annot=True)
    columnNames=list(df.columns)
    session['columnNames']=columnNames
    
   

    return render_template('dashboard.html',fileName=session.get('fileName'), head=dfHead
    , tail=dfTail,info=dfInfo, summary=dfSummary, nullValue=nullValue,  )

@app.route('/operations')
def operations():
    return render_template('operations.html')


@app.route('/processData', methods=['GET', 'POST'])
def processData():
    # Load the uploaded file content from the session if it's not already loaded
    uploaded_file_content = session.get('uploaded_file')
    
    if not uploaded_file_content:
        return render_template('error.html', message="No file uploaded yet!")

    # Initially load the data into a DataFrame from the session
    df = pd.read_csv(io.StringIO(uploaded_file_content))
    
    # If a modified DataFrame already exists in the session, use it instead
    processed_df_content = session.get('processed_df')
    if processed_df_content:
        df = pd.read_csv(io.StringIO(processed_df_content))
     # Use the latest modified DataFrame

    if request.method == 'POST':
        # Perform operations based on the button pressed
        if 'remove_null' in request.form:    
           df.dropna(inplace=True)  # Remove rows with null values
            
        if 'columnNames' in request.form:
            columns_to_remove = request.form.getlist('columnNames')  # Get selected columns
            df.drop(columns=columns_to_remove, inplace=True, errors='ignore')  # Drop selected columns
        if 'label_encode' in request.form:
            df = label_encode_categorical_columns(df)  # Apply label encoding to categorical columns

        # Save the updated DataFrame to session after each operation
        session['processed_df'] = df.to_csv(index=False)

        # Prepare summary and other data for display
        nullValue = df.isnull().sum().to_frame(name='Null Count').to_html()
        dfHead = df.head().to_html()
        columnNames = list(df.columns)


        # Re-render the process data page with updated data
        return render_template('processData.html', nullValue=nullValue, head=dfHead, columnNames=columnNames)

    else:
        # For GET requests (first load), load the data from session or file
        nullValue = df.isnull().sum().to_frame(name='Null Count').to_html()
        dfHead = df.head().to_html()
        columnNames = list(df.columns)

        return render_template('processData.html', nullValue=nullValue, head=dfHead, columnNames=columnNames)


@app.route("/task")
def task():
    df=pd.read_csv(io.StringIO(session.get('processed_df')))
    nullValue = df.isnull().sum().to_frame(name='Null Count').to_html()
    dfHead = df.head().to_html()

    return render_template('task.html', nullValue=nullValue, head=dfHead)

@app.route('/under_construction')
def under_construction():
    return render_template('underconst.html')



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    #app.run(debug=True)