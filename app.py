from flask import Flask,session ,render_template, request, redirect, url_for
from flask_session import Session
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import io
import os



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
   

    return render_template('dashboard.html',fileName=session.get('fileName'), head=dfHead
    , tail=dfTail,info=dfInfo, summary=dfSummary, nullValue=nullValue )

@app.route('/operations')
def operations():
    return render_template('operations.html')


@app.route('/nullvalue' , methods=['GET', 'POST'])
def nullvalue():

    if request.method == 'POST':
        df=pd.read_csv(io.StringIO(session.get('uploaded_file')))
        df.dropna(inplace=True)
        nullValue = df.isnull().sum().to_frame(name='Null Count').to_html()
        encoded_df=label_encode_categorical_columns(df)
        dfHead=encoded_df.head().to_html()
        session['encoded_df'] = encoded_df
        
        return render_template('task.html',head=dfHead, nullValue=nullValue )
    
    
    
    else:
        return render_template('handelNull.html', nullvalue=session.get('null_count'), )
    

@app.route('/under_construction')
def under_construction():
    return render_template('underconst.html')



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)