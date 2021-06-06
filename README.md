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
  * disaster_categories.csv and disaster_messages.csv (dataset)
  * DisasterResponse.db: created database from transformed and cleaned data.
* Model
  * train_classifier.py: includes the code necessary to load data, transform data for NLP, and train a machine learning classifier using GridSearchCV. Basic usage is python train_classifier.py DATABASE_DIRECTORY SAVENAME_FOR_MODEL
* App
  * run.py: Flask app and the user interface used to predict results and display them.
  * templates: folder containing the html templates
## How to Run <a name="howto"></a>

## Results <a name="results"></a> 

## Licensing, Authors, and Acknowledgements<a name="licensing"></a> 
Credit for the data goes to Figure Eight and Udacity
