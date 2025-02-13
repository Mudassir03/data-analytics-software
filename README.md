# Machine Learning Model Selection Web Application

## Overview
This web application allows users to upload a CSV dataset, preprocess the data, and automatically determine the best-performing machine learning model for regression or classification tasks using Grid Search CV.

## Features
- Upload a CSV file and view dataset details (head, tail, summary, and null values).
- Perform preprocessing operations like removing null values, dropping columns, and encoding categorical data.
- Select between regression or classification tasks.
- Choose the target variable for prediction.
- Identify the best-performing algorithm with optimized hyperparameters using Grid Search CV.
- Download the processed dataset.

## Tech Stack
- **Backend:** Flask, Pandas, NumPy, Scikit-Learn
- **Frontend:** HTML, CSS, Bootstrap
- **Deployment:** AWS EC2 Instance

## Project Flow
1. **File Upload:** Users upload a CSV file, which is validated and stored in the session.
2. **Data Overview:** Displays dataset details like head, tail, summary, and null values.
3. **Preprocessing:** Users can remove null values, drop columns, and encode categorical data.
4. **Task Selection:** Users select regression or classification and define the target variable.
5. **Model Selection:** The app determines the best-performing model using Grid Search CV.
6. **Results Display:** Shows the best model, evaluation metrics, and hyperparameters.
7. **Download Processed Data:** Users can download the cleaned dataset for further use.

## Challenges Faced
- Initially, operations were performed directly on the original dataset, causing loss of data. A separate processed dataset was introduced to resolve this.
- The application mistakenly retained previous session data when a new file was uploaded. This was fixed by clearing session data upon file upload.
- If a dataset was already clean, users could skip preprocessing, leading to session errors. A filter was added to handle this case.

## Deployment
The application was successfully deployed on AWS using an EC2 instance.

## How to Run Locally
1. Clone the repository:
   ```sh
   git clone <repository_url>
   ```
2. Navigate to the project folder:
   ```sh
   cd project-folder
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   python app.py
   ```
5. Open a web browser and go to:
   ```
   http://127.0.0.1:8080/
   ```

## Contact
**Developer:** Mudassir Khan  
📧 Email: [khanmudassir.work@gmail.com](mailto:khanmudassir.work@gmail.com)

If you are interested in contributing to this project, feel free to reach out to me. I would be happy to collaborate and discuss any improvements or suggestions.

