from flask import Flask,session ,render_template, request, redirect, url_for
from flask_session import Session
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import io
import os
import seaborn as sb



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier






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


#regression model selection

def best_regression_model_with_grid_search(X, y):
    """
    Perform grid search to tune hyperparameters for multiple regression models 
    and return the best model based on R^2 score.

    Parameters:
    X (DataFrame or ndarray): The feature matrix.
    y (Series or ndarray): The target variable.

    Returns:
    dict: Best model, R^2 score, and other metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Support Vector Regressor': SVR(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor()
    }

    param_grids = {
        'Linear Regression': {},
        'Ridge Regression': {'alpha': [0.1, 1, 10, 100]},
        'Lasso Regression': {'alpha': [0.1, 1, 10, 100]},
        'Support Vector Regressor': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'Random Forest Regressor': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'Gradient Boosting Regressor': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}
    }

    results = {}

    for model_name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], 
                                   scoring='r2', cv=5, n_jobs=-1, verbose=2)
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        results[model_name] = {
            'Model': best_model,
            'Best Hyperparameters': grid_search.best_params_,
            'R²': r2,
            'MSE': mse,
            'MAE': mae
        }

    best_model_name = max(results, key=lambda x: results[x]['R²'])
    best_model = results[best_model_name]
    
    return best_model


#classification model selection




def best_classification_model_with_grid_search(X, y):
    """
    Perform grid search to tune hyperparameters for multiple classification models 
    and return the best model based on accuracy score.

    Parameters:
    X (DataFrame or ndarray): The feature matrix.
    y (Series or ndarray): The target variable.

    Returns:
    dict: Best model, accuracy score, and other metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Support Vector Classifier': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting Classifier': GradientBoostingClassifier()
    }

    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
        'Support Vector Classifier': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'Gradient Boosting Classifier': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}
    }

    results = {}

    for model_name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], 
                                   scoring='accuracy', cv=5, n_jobs=-1 , verbose=2)
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[model_name] = {
            'Model': best_model,
            'Best Hyperparameters': grid_search.best_params_,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

    best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
    best_model = results[best_model_name]
    
    return best_model





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

        if 'reset' in request.form:
            # Clear the session of the processed DataFrame and reload the original file
            session.pop('processed_df', None)
            df = pd.read_csv(io.StringIO(uploaded_file_content))  # Reload the original DataFrame
            nullValue = df.isnull().sum().to_frame(name='Null Count').to_html()
            dfHead = df.head().to_html()
            columnNames = list(df.columns)
            # Re-render the page with the reset data
            return render_template('processData.html', nullValue=nullValue, head=dfHead, columnNames=columnNames)
        if 'clear_session' in request.form:
            # Clear all session data
            session.clear()  # This will clear all session variables
            # Redirect to home page or upload page
            return redirect(url_for('upload_file'))
        # Perform operations based on the button pressed
        if 'remove_null' in request.form:    
           df.dropna(inplace=True)  # Remove rows with null values
            
        if 'columnNames' in request.form:
            columns_to_remove = request.form.getlist('columnNames')  # Get selected columns
            if len(df.columns) - len(columns_to_remove) < 2:
                return render_template('processData.html', error="Cannot remove columns. At least two columns must remain.", nullValue=df.isnull().sum().to_frame(name='Null Count').to_html(), head=df.head().to_html(), columnNames=df.columns.tolist())
              # Drop selected columns
            df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
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

    if not session.get('processed_df'):
        if not  session.get('uploaded_file'):
            return render_template('error.html', message="No uploaded file found in the session!")
        else:
           session['processed_df']= (session.get('uploaded_file'))
    df=pd.read_csv(io.StringIO(session.get('processed_df')))
    

    if (df.isnull().values.any()) or (len(df.select_dtypes(include=['object', 'category']).columns) > 0):
        return render_template('baddata.html', message="The data contains missing values or categorical variables. Please preprocess the data before performing Machine Learning tasks.")


    
            

    


    
    nullValue = df.isnull().sum().to_frame(name='Null Count').to_html()
    dfHead = df.head().to_html()

    return render_template('task.html', nullValue=nullValue, head=dfHead)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if not session.get('processed_df'):
        return render_template('error.html', message="No data available for regression!")

    df = pd.read_csv(io.StringIO(session.get('processed_df')))
    
    if request.method == 'POST':
        selected_columns = request.form.getlist('columnNames')
        
        if not selected_columns or len(selected_columns) != 1:
            return render_template('error.html', message="Please select exactly one target variable!")
        
        y = df[selected_columns[0]]  # Dependent variable
        X = df.drop(columns=selected_columns[0])  # Independent variables
        
        best_model_info = best_regression_model_with_grid_search(X, y)
        
        # Extract details
        model_name = type(best_model_info['Model']).__name__
        r2 = best_model_info['R²']
        mse = best_model_info['MSE']
        mae = best_model_info['MAE']
        best_hyperparameters = best_model_info['Best Hyperparameters']
        
        return render_template(
            'rfinalStage.html', 
            model_name=model_name, r2=r2, mse=mse, mae=mae, best_hyperparameters=best_hyperparameters
        )
    else:
        columnNames = list(df.columns)
        return render_template('regression.html', columnNames=columnNames)
    
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if not session.get('processed_df'):
        return render_template('error.html', message="No data available for classification!")

    df = pd.read_csv(io.StringIO(session.get('processed_df')))

    if request.method == 'POST':
        selected_columns = request.form.getlist('columnNames')

        if not selected_columns or len(selected_columns) != 1:
            return render_template('error.html', message="Please select exactly one target variable!")

        y = df[selected_columns[0]]  # Dependent variable
        X = df.drop(columns=selected_columns[0])  # Independent variables

        best_model_info = best_classification_model_with_grid_search(X, y)

        # Extract details
        model_name = type(best_model_info['Model']).__name__
        accuracy = best_model_info['Accuracy']
        precision = best_model_info['Precision']
        recall = best_model_info['Recall']
        f1_score = best_model_info['F1 Score']
        best_hyperparameters = best_model_info['Best Hyperparameters']

        return render_template(
            'cfinalStage.html', 
            model_name=model_name, 
            accuracy=accuracy, 
            precision=precision, 
            recall=recall, 
            f1_score=f1_score, 
            best_hyperparameters=best_hyperparameters
        )
    else:
        columnNames = list(df.columns)
        return render_template('classification.html', columnNames=columnNames)
    
@app.route('/terminate')    
def terminate():
    return redirect(url_for('upload_file'))
    # This will clear all session variables
            # Redirect to home page or upload page
            




@app.route('/under_construction')
def under_construction():
    return render_template('underconst.html')



if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=8080)
    app.run(debug=True)