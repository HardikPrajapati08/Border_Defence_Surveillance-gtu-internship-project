import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='object_class', palette='Set2', ax=ax)
    ax.set_title('Detection Frequency by Default Class')
    ax.set_xlabel('Object Class')
    ax.set_ylabel('Count')
    return fig

def plot_threat_vs_normal(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Threat level is 0 or 1
    counts = df['threat_level'].value_counts()
    ax.pie(counts, labels=['Normal (0)', 'Threat/Anomaly (1)'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
    ax.set_title('Normal Activity vs Anomalous Threats')
    return fig

def plot_time_of_day_threats(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df[df['threat_level'] == 1], x='time_of_day', bins=24, kde=True, color='red', ax=ax)
    ax.set_title('Threat Detections Over 24 Hours')
    ax.set_xlabel('Time of Day (Hour)')
    ax.set_ylabel('Number of Threats')
    ax.set_xticks(range(0, 24))
    return fig

def plot_spatial_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    threats = df[df['threat_level'] == 1]
    
    if len(threats) > 0:
        sns.kdeplot(
            data=threats, x="x_coord", y="y_coord", 
            fill=True, cmap="YlOrRd", alpha=0.8, ax=ax
        )
        ax.scatter(threats['x_coord'], threats['y_coord'], color='black', marker='x', s=10, alpha=0.5, label='Anomaly Point')
        ax.legend()
    
    ax.set_title('Spatial Vulnerability Heatmap (High Risk Zones)')
    ax.set_xlabel('X Coordinate (Border Longitude map)')
    ax.set_ylabel('Y Coordinate (Border Latitude map)')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    return fig
