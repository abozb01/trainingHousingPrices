# trainingHousingPrices
Machine learning
# Housing Price Prediction Tool

This Python-based tool uses machine learning to predict housing prices based on various features. It offers a graphical user interface (GUI) built using Tkinter, which allows users to load datasets, visualize data, analyze correlations, and train a machine learning model using a Random Forest Regressor.

## Features

- **Load Data**: Load the default dataset or provide a custom dataset URL.
- **Data Cleaning**: Automatically handles missing values and converts columns to appropriate data types.
- **Visualize Missing Values**: Shows a bar chart of the percentage of missing values for each column.
- **Visualize Correlations**: Displays a correlation matrix heatmap and allows saving the matrix to a CSV file.
- **Visualize Data**: Provides an interface to select columns for plotting scatter plots.
- **Train Model**: Trains a Random Forest Regressor on the dataset and evaluates the model's performance using Mean Absolute Error (MAE).

## Installation

To run this program locally, you'll need to have Python 3.x installed along with the required dependencies. You can install them using the following command:

```bash
pip install -r requirements.txt
