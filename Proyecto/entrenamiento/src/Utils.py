import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def shifted_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return mean_absolute_percentage_error(y_true + 1.0, y_pred + 1.0)


def plot_performance_analysis(y_true, y_pred, variable_name, unit_label, path=None):
    epsilon = 1e-10
    pct_error = ((y_true - y_pred) / (y_true + epsilon)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(pct_error, bins=100, range=(-40, 40), color='#1f77b4', edgecolor='none')
    axes[0].set_xlabel('Error (%)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title(f'Distribucion del Error Porcentual - {variable_name}')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_true, y_pred, alpha=0.6, color='blue', s=20, label='Predicciones')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r-', lw=2, label='Ideal')

    axes[1].set_xlabel(f'Valores reales: {variable_name} ({unit_label})')
    axes[1].set_ylabel(f'Predicciones: {variable_name} ({unit_label})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()
    if path is not None:
        fig.savefig(path)

def mse_column_score(y_true, y_pred, col_idx):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return mean_squared_error(y_true[:, col_idx], y_pred[:, col_idx])


def plot_params_vs_mse(results, target_error_col, estimator_prefix, param_cols, title):
    est_params = [c for c in param_cols if estimator_prefix in c]

    n_cols = 2
    n_rows = math.ceil(len(est_params) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Impacto de Hiperparámetros en {title}', fontsize=16)
    axes = axes.flatten()

    for i, param in enumerate(est_params):
        x_data = results[param].astype(float)
        y_data = -results[target_error_col]

        sns.regplot(x=x_data, y=y_data, ax=axes[i], scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})

        best_idx = y_data.idxmin()
        axes[i].scatter(x_data[best_idx], y_data[best_idx], c='green', s=100, label='Mejor modelo', edgecolors='black')

        axes[i].set_title(param.replace('param_', ''))
        axes[i].set_xlabel('Valor del Parámetro')
        axes[i].set_ylabel('Error (MSE)')
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_histograms(df, title, path):
    sns.set_theme(style="whitegrid")

    n_columnas = len(df.columns)
    cols_grid = 5
    rows_grid = math.ceil(n_columnas / cols_grid)

    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(16, 4 * rows_grid))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        sns.histplot(data=df, x=col, kde=False, ax=axes[i], bins=30,
                     color='blue', edgecolor='black', alpha=0.5, stat="density")

        sns.kdeplot(data=df, x=col, ax=axes[i], color='red', linewidth=2)

        axes[i].set_title(f'Distribución de {col}', color='black')
        axes[i].set_xlabel('', color='black')
        axes[i].set_ylabel('Densidad' if i % cols_grid == 0 else '', color='black')
        axes[i].tick_params(axis='both', colors='black')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    if path is not None:
        fig.savefig(path)