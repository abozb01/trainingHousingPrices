# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 09:02:30 2025

@author: abozy
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Default data URL
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/datasets/house-prices-us/main/data/houses.csv"

# Data cleaning functions
def clean_data(df):
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Convert columns to appropriate data types if needed
    if 'LotArea' in df.columns:
        df['LotArea'] = df['LotArea'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else x)

    # Fill missing values with median for numerical and mode for categorical columns
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# GUI Functions
def load_data_from_url(url):
    global df
    try:
        df = pd.read_csv(url)
        df = clean_data(df)
        messagebox.showinfo("Success", "Data loaded and cleaned successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")

def load_custom_data():
    input_window = tk.Toplevel(root)
    input_window.title("Enter Dataset URL")

    tk.Label(input_window, text="Enter Dataset URL:").pack(pady=5)
    url_entry = tk.Entry(input_window, width=50)
    url_entry.pack(pady=5)

    def submit_url():
        url = url_entry.get()
        if url:
            load_data_from_url(url)
            input_window.destroy()
        else:
            messagebox.showwarning("Warning", "Please enter a valid URL.")

    tk.Button(input_window, text="Submit", command=submit_url).pack(pady=10)

def load_default_data():
    load_data_from_url(DEFAULT_DATA_URL)

def show_info():
    if df is not None:
        info_window = tk.Toplevel(root)
        info_window.title("Dataset Info")

        # Dataset description
        text = tk.Text(info_window, wrap=tk.WORD, width=100, height=30)
        text.insert(tk.END, str(df.describe()))
        text.insert(tk.END, "\n\n")
        text.insert(tk.END, str(df.info()))
        text.pack()
    else:
        messagebox.showwarning("Warning", "Load the data first!")

def visualize_missing_values():
    if df is not None:
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        filtered_missing_percentage = missing_percentage[missing_percentage > 0]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=filtered_missing_percentage.index, y=filtered_missing_percentage.values)
        plt.title("Missing Values Percentage")
        plt.xticks(rotation=45)
        plt.ylabel("Percentage")
        plt.xlabel("Columns")
        plt.show()
    else:
        messagebox.showwarning("Warning", "Load the data first!")

def visualize_correlations():
    if df is not None:
        correlation_matrix = df.corr()

        # Show heatmap in a new window
        def save_correlations():
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                correlation_matrix.to_csv(file_path)
                messagebox.showinfo("Success", "Correlation matrix saved successfully!")

        correlation_window = tk.Toplevel(root)
        correlation_window.title("Correlation Matrix")

        tk.Label(correlation_window, text="Correlation Matrix", font=("Arial", 14)).pack(pady=10)
        save_button = tk.Button(correlation_window, text="Save as CSV", command=save_correlations)
        save_button.pack(pady=5)

        # Plot the heatmap in a new window
        def show_heatmap():
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.show()

        heatmap_button = tk.Button(correlation_window, text="Show Heatmap", command=show_heatmap)
        heatmap_button.pack(pady=5)
    else:
        messagebox.showwarning("Warning", "Load the data first!")

def visualize_data():
    if df is not None:
        visualize_window = tk.Toplevel(root)
        visualize_window.title("Visualize Data")

        tk.Label(visualize_window, text="Select Column for X-axis:").pack(pady=5)
        x_column = ttk.Combobox(visualize_window, values=list(df.columns))
        x_column.pack(pady=5)

        tk.Label(visualize_window, text="Select Column for Y-axis:").pack(pady=5)
        y_column = ttk.Combobox(visualize_window, values=list(df.columns))
        y_column.pack(pady=5)

        def plot_graph():
            x_col = x_column.get()
            y_col = y_column.get()
            if x_col and y_col:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=x_col, y=y_col)
                plt.title(f"Scatter Plot: {x_col} vs {y_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.show()
            else:
                messagebox.showwarning("Warning", "Please select both X and Y columns.")

        tk.Button(visualize_window, text="Plot", command=plot_graph).pack(pady=10)
    else:
        messagebox.showwarning("Warning", "Load the data first!")

def train_model():
    if df is not None:
        try:
            # Select features and target (assuming 'SalePrice' is the target variable)
            features = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice'], errors='ignore')
            target = df['SalePrice']

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # Train Random Forest model
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            messagebox.showinfo("Model Performance", f"Mean Absolute Error: {mae:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train the model: {e}")
    else:
        messagebox.showwarning("Warning", "Load the data first!")

# Main GUI
root = tk.Tk()
root.title("Housing Price Prediction")
root.geometry("400x500")

tk.Button(root, text="Load Default Data", command=load_default_data).pack(pady=10)
tk.Button(root, text="Load Custom Data", command=load_custom_data).pack(pady=10)
tk.Button(root, text="Show Info", command=show_info).pack(pady=10)
tk.Button(root, text="Visualize Missing Values", command=visualize_missing_values).pack(pady=10)
tk.Button(root, text="Visualize Correlations", command=visualize_correlations).pack(pady=10)
tk.Button(root, text="Visualize Data", command=visualize_data).pack(pady=10)
tk.Button(root, text="Train Model", command=train_model).pack(pady=10)

root.mainloop()
