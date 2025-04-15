"""
visualization.py - visualization utilities

This module offers a collection of visualization utilities
for data analysis and machine learning model evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

def plot_yield_trends(yield_data, regions, output_dir=None, filename="yield_trends.png"):
    """
    Visualize yield trends by region
    
    Args:
        yield_data (dict): dict of yield data by region
        regions (list): list of regions
        output_dir (str, optional)
        filename (str)
    """
    plt.figure(figsize=(14, 7))
    
    for region in regions:
        if region in yield_data:
            # Year/Yearカラム名の違いを吸収
            year_col = 'Year' if 'Year' in yield_data[region].columns else 'year'
            yield_col = 'Yields' if 'Yields' in yield_data[region].columns else 'yields'
            
            plt.plot(yield_data[region][year_col], 
                     yield_data[region][yield_col], 
                     marker='o', linewidth=2, label=region)

    plt.title('Yield Trends by Region', fontsize=16)
    plt.xlabel('年', fontsize=14)
    plt.ylabel('Yield (kg/ha)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved yield trends plot: {output_path}")
    
    plt.show()
    plt.close()

def plot_feature_correlations(data, regions, feature_cols, output_dir=None, filename="feature_correlations.png"):
    """
    Visualize feature correlations
    
    Args:
        data (pd.DataFrame): DataFrame of features and target variable
        regions (list)
        feature_cols (list): list of columns to include in correlation analysis
        output_dir (str, optional)
        filename (str)
    """
    # Identify yield column
    yield_col = 'Yields' if 'Yields' in data.columns else 'yields'
    
    # Columns for correlation analysis
    correlation_cols = feature_cols + [yield_col]
    
    plt.figure(figsize=(18, 14))
    
    # Analyze each region
    for i, region in enumerate(regions):
        region_data = data[data['region'] == region]
        
        if len(region_data) > 0:
            corr_matrix = region_data[correlation_cols].corr()
            
            plt.subplot(2, 2, i+1)
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                        fmt='.2f', linewidths=0.5, cbar=True)
            plt.title(f'{region}の気象要因と収穫量の相関', fontsize=14)
            plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout(pad=3.0)
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"特徴量相関図を保存: {output_path}")
    
    plt.show()
    plt.close()

def plot_scatter_by_region(data, x_col, y_col='Yields', regions=None, output_dir=None, filename="scatter_by_region.png"):
    """
    Scatter plot by region
    
    Args:
        data (pd.DataFrame)
        x_col (str)
        y_col (str)
        regions (list, optional)
        output_dir (str, optional)
        filename (str)
    """
    if regions is None:
        regions = data['region'].unique()
    
    plt.figure(figsize=(16, 10))
    
    for i, region in enumerate(regions):
        region_data = data[data['region'] == region]
        
        if len(region_data) > 0:
            plt.subplot(2, 2, i+1)
            
            sns.regplot(x=x_col, y=y_col, data=region_data, 
                        scatter_kws={'s': 80, 'alpha': 0.7}, 
                        line_kws={'color': 'red', 'linewidth': 2})
            
            plt.title(f'{region}: {x_col}と{y_col}の関係', fontsize=14)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved scatter plot: {output_path}")
    
    plt.show()
    plt.close()

def plot_feature_importance(model, feature_names, output_dir=None, filename="feature_importance.png"):
    """
    Visualize feature importance
    
    Args:
        model: trained model (coefficients_ or feature_importances_)
        feature_names (list)
        output_dir (str, optional)
        filename (str)
    """
    # Acquire feature importance based on model type
    if hasattr(model, 'coef_'):
        # Linear model
        importance = model.coef_
        title = 'Feature Importance (Coefficients)'
        xlabel = 'Coef. Value'
    elif hasattr(model, 'feature_importances_'):
        # Tree model
        importance = model.feature_importances_
        title = 'Feature Importance'
        xlabel = 'Importance'
    else:
        logger.error("Failed to acquire feature importance.")
        return
    
    # Store feature importance to dataframe
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    coefficients = coefficients.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=coefficients)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved feature importance: {output_path}")
    
    plt.show()
    plt.close()

def plot_predictions(y_true, y_pred, output_dir=None, filename="predictions.png"):
    """
    Plot predictions vs. actual values
    
    Args:
        y_true (array-like): actual values
        y_pred (array-like): predicted values
        output_dir (str, optional)
        filename (str)
        
    Returns:
        dict: metrics（RMSE, R²）
    """
    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # Draw diagonal line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted', fontsize=16)
    plt.xlabel('Actual Yields (kg/ha)', fontsize=14)
    plt.ylabel('Pred. Yields (kg/ha)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot evaluation metrics
    plt.annotate(f"RMSE: {rmse:.2f}\nR²: {r2:.4f}", 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=12)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved plots: {output_path}")
    
    plt.show()
    plt.close()
    
    return {'rmse': rmse, 'r2': r2}

def plot_regional_boxplots(data, y_col='Yields', output_dir=None, filename="regional_boxplots.png"):
    """
    Visualize yields by region with boxplot
    
    Args:
        data (pd.DataFrame)
        y_col (str)
        output_dir (str, optional)
        filename (str)
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='region', y=y_col, data=data)
    plt.title('Plots of Yields by Region', fontsize=16)
    plt.xlabel('Region', fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved boxplots: {output_path}")
    
    plt.show()
    plt.close()

def plot_residuals(y_true, y_pred, output_dir=None, filename="residuals.png"):
    """
    Residual plot
    
    Args:
        y_true (array-like): actual values
        y_pred (array-like): predicted values
        output_dir (str, optional)
        filename (str)
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 6))
    
    # Left: Pred. vs. Residual
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Predicted vs Residual', fontsize=14)
    plt.xlabel('Pred. Values', fontsize=12)
    plt.ylabel('Residual', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 右側: 残差の分布
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution', fontsize=14)
    plt.xlabel('Residual', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved residual plot: {output_path}")
    
    plt.show()
    plt.close()

def plot_growing_stages_comparison(climate_data, region, year, output_dir=None, filename="growing_stages.png"):
    """
    Visualize climate data for specific year and region by stage
    
    Args:
        climate_data (pd.DataFrame)
        region (str)
        year (int)
        output_dir (str, optional)
        filename (str)
    """
    # Define stages
    stages = {
        'preparation': [1, 2, 3, 4],
        'planting': [5],
        'growing': [6, 7],
        'heading': [8, 9],
        'harvesting': [10]
    }
    
    # filter by year and region
    year_data = climate_data[(climate_data['year'] == year) & (climate_data['city'] == region)]
    
    if len(year_data) == 0:
        logger.warning(f"Year ({year}) and Region ({region}) data not found. Skipping...")
        return
    
    # Aggregate data by month
    monthly_avg = year_data.groupby('month').agg({
        'temp_2m': 'mean',
        'soil_temp_l1': 'mean',
        'total_rain': 'sum'
    }).reset_index()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot temp (left: temp, right: soil temp)
    ax1.plot(monthly_avg['month'], monthly_avg['temp_2m'], 'o-', color='red', linewidth=2, label='Temperature')
    ax1.set_ylabel('Temp (K)', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax1b = ax1.twinx()
    ax1b.plot(monthly_avg['month'], monthly_avg['soil_temp_l1'], 'o--', color='brown', linewidth=2, label='Soil Temperature')
    ax1b.set_ylabel('Soil Temp (K)', color='brown', fontsize=12)
    ax1b.tick_params(axis='y', labelcolor='brown')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot rain (right: total rain)
    ax2.bar(monthly_avg['month'], monthly_avg['total_rain'], color='blue', alpha=0.7, label='Precipitation')
    ax2.set_xlabel('Month', fontsize=14)
    ax2.set_ylabel('Total Rain (m)', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper left')
    
    # Axis settings
    ax1.set_xlim(0.5, 12.5)
    ax2.set_xlim(0.5, 12.5)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    
    # Stage colors
    stage_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lavender']
    
    for i, (stage_name, months) in enumerate(stages.items()):
        # Stage boundaries
        min_month = min(months) - 0.5
        max_month = max(months) + 0.5
        
        # Bg color
        ax1.axvspan(min_month, max_month, alpha=0.2, color=stage_colors[i])
        ax2.axvspan(min_month, max_month, alpha=0.2, color=stage_colors[i])
        
        # Stage name on top
        ax1.text((min_month + max_month) / 2, ax1.get_ylim()[1] * 0.95, 
                 stage_name, ha='center', va='top', fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title(f'Monthly Climate Data by Stages for {region} in {year}', fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved climate plot: {output_path}")
    
    plt.show()
    plt.close()