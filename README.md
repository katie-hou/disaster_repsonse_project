## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [How to Run](#howto)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code should run with no issues using Python versions 3, without any extra library beyond the Anaconda distribution of Python required.
## Project Motivation <a name="motivation"></a> 
The purpose of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages into different categories.
## File Description <a name="files"></a>
* Data
  * process_data.py: reads in the data, cleans and stores it in a SQL database. 
  * disaster_categories.csv: categories dataset
  * disaster_messages.csv: messages dataset
  * DisasterResponse.db: created database after loading and cleaning the data
* Models
  * train_classifier.py: includes the code necessary to load data, transform data for NLP, and train a machine learning classifier using GridSearchCV.
  * classifier.pkl: saved classifier pickle file
* App
  * run.py: Flask app and the user interface used to predict results and display them.
  * templates: folder containing the html templates
## How to Run <a name="howto"></a>
below are sample bash commands to run the ETL, model, and app in terminal:

'python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'

'python train_classifier.py ../data/DisasterResponse.db classifier.pkl'

'python run.py'
## Results <a name="results"></a> 
screeshot of the UI to predict results:
![alt text](https://github.com/katie-hou/disaster_response_project/blob/main/ss1.png)

screeshot of the UI to display data visualizations:
![alt text](https://github.com/katie-hou/disaster_response_project/blob/main/ss2.png)
## Licensing, Authors, and Acknowledgements<a name="licensing"></a> 
Credit for the data goes to Figure Eight and Udacity.
