import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.metrics import roc_curve, auc

# Set consistent style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

class ModelVisualizer:

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.colors = None
        
    def _setup_plot(self, figsize: tuple = (12, 8)) -> tuple:
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        return fig, ax
    
    def _save_plot(self, filename: str):
        plt.savefig(f'{self.save_path}/{filename}.png', 
                   bbox_inches='tight', 
                   dpi=300,
                   facecolor='white')
        plt.close()
    
    def _get_colors(self, n_models: int):
        if self.colors is None or len(self.colors) != n_models:
            self.colors = sns.color_palette("husl", n_models)
        return self.colors

    def plot_runtime_comparison(self, results: Dict, train_sizes: list):
        _, ax = self._setup_plot()
        colors = self._get_colors(len(results))
        line_width = 2
        
        max_time = float('-inf')
        min_time = float('inf')
        
        for (model_name, color) in zip(results.keys(), colors):
            mean_times = [results[model_name][size]['mean']['TrainTime'] for size in train_sizes]
            var_times = [results[model_name][size]['var']['TrainTime'] for size in train_sizes]
            
            means = np.array(mean_times)
            stds = np.sqrt(np.array(var_times))
            
            # Create smoother curves
            x_smooth = np.linspace(min(train_sizes), max(train_sizes), 200)
            means_smooth = scipy.interpolate.make_interp_spline(train_sizes, means)(x_smooth)
            stds_smooth = scipy.interpolate.make_interp_spline(train_sizes, stds)(x_smooth)
            
            # Plot 2-sigma band (lighter)
            ax.fill_between(x_smooth,
                        means_smooth - 2*stds_smooth,
                        means_smooth + 2*stds_smooth,
                        color=color,
                        alpha=0.1)
            
            # Plot 1-sigma band (darker)
            ax.fill_between(x_smooth,
                        means_smooth - stds_smooth,
                        means_smooth + stds_smooth,
                        color=color,
                        alpha=0.2)
            
            # Plot mean line
            ax.plot(x_smooth, means_smooth,
                marker='',
                color=color,
                linewidth=line_width,
                label=model_name,
                linestyle='--')
            
            # Add points for actual data
            ax.scatter(train_sizes, means,
                    color=color,
                    s=50,
                    zorder=5)
            
            max_time = max(max_time, np.max(means + 2*stds))
            min_time = min(min_time, np.min(means - 2*stds))
        
        ax.set_xlabel('Training Set Size', fontsize=16, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=16, fontweight='bold')
        ax.set_title('Runtime Comparison Across Models', fontsize=18, pad=20)
        
        if max_time / min_time > 100:
            ax.set_yscale('log')
        
        ax.legend(title='Models',
                title_fontsize=16,
                fontsize=14,
                bbox_to_anchor=(1.05, 1),
                loc='upper left')
        
        ax.grid(True, alpha=0.7)
        self._save_plot('runtime_comparison')

    def plot_metrics_analysis(self, results: Dict, train_sizes: list):

        fig, axes = plt.subplots(1, 3, figsize=(20, 20))
        axes = axes.flatten()
        colors = self._get_colors(len(results))
        metrics = ['MSE', 'MAE', 'RMAE']
        
        # Define different line styles
        line_styles = {
            'S-Learner': '-',
            'T-Learner': '-',
            'X-Learner': '-.',
            'DR-Learner': '-',
            'Causal-Forest': '-'
        }
        
        for idx, (model_name, model_results) in enumerate(results.items()):
            mean_values = {metric: [] for metric in metrics}
            var_values = {metric: [] for metric in metrics}
            
            for size in train_sizes:
                for metric in metrics:
                    mean_values[metric].append(model_results[size]['mean'][metric])
                    var_values[metric].append(model_results[size]['var'][metric])
            
            for ax, metric in zip(axes, metrics):
                means = np.array(mean_values[metric])
                stds = np.sqrt(np.array(var_values[metric]))
            
                ax.fill_between(train_sizes,
                            means - 2*stds,
                            means + 2*stds,
                            color=colors[idx],
                            alpha=0.1)
                
                ax.fill_between(train_sizes,
                            means - stds,
                            means + stds,
                            color=colors[idx],
                            alpha=0.2)
                
                ax.plot(train_sizes, means,
                    marker='o',
                    label=model_name,
                    linewidth=2,
                    color=colors[idx],
                    linestyle=line_styles[model_name])
                
                ax.scatter(train_sizes, means,
                        color=colors[idx],
                        s=50,
                        zorder=5)
                
                ax.set_xlabel('Training Size', fontsize=16, fontweight='bold', labelpad=10)
                ax.set_ylabel(metric, fontsize=16, fontweight='bold', labelpad=10)
                ax.set_title(f'{metric} vs Training Size', fontsize=18, fontweight='bold', pad=15)
                ax.grid(True, alpha=1)
                ax.tick_params(axis='both', which='major', labelsize=14)
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                title='Models',
                bbox_to_anchor=(0.5, 0.05),
                loc='center',
                ncol=len(results),
                fontsize=26,
                title_fontsize=28)
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        self._save_plot('metrics_analysis')

    def plot_fairness(self, results: Dict, train_sizes: list):
        _, ax = self._setup_plot()
        colors = self._get_colors(len(results) + 1)
        line_width = 2
        metrics = ["fairness"]

        for idx, (model_name, model_results) in enumerate(results.items()):
            mean_values = {metric: [] for metric in metrics}
            
            for size in train_sizes:
                for metric in metrics:
                    mean_values[metric].append(model_results[size]['mean'][metric])

            means = np.array(mean_values[metric])
            
            # Plot line connecting points
            ax.plot(train_sizes, means,
                    color=colors[idx + 1],
                    linewidth=line_width)
            
            # Plot scatter points on top of line
            ax.scatter(train_sizes, means,
                    color=colors[idx + 1],
                    s=50,
                    zorder=5,
                    label=model_name)
                
                
        ax.set_xlabel('Training Size', fontsize=16, fontweight='bold')
        ax.set_ylabel('ΔPF', fontsize=16, fontweight='bold')
        #ax.set_title(f'{metric} vs Training Size', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.legend(title='Models',
                title_fontsize=16,
                fontsize=14,
                loc='upper right',
                framealpha=0.9)
            
        #plt.tight_layout()
        self._save_plot("fairness")


    def plot_cate_mse(self, results: Dict, train_sizes: list):
        _, ax = self._setup_plot()
        colors = self._get_colors(len(results) + 1)
        line_width = 2
        metrics = ["cate_mse"]

        for idx, (model_name, model_results) in enumerate(results.items()):
            mean_values = {metric: [] for metric in metrics}
            
            for size in train_sizes:
                for metric in metrics:
                    mean_values[metric].append(model_results[size]['mean'][metric])

            means = np.array(mean_values[metric])
            
            # Plot line connecting points
            ax.plot(train_sizes, means,
                    color=colors[idx + 1],
                    linewidth=line_width)
            
            # Plot scatter points on top of line
            ax.scatter(train_sizes, means,
                    color=colors[idx + 1],
                    s=50,
                    zorder=5,
                    label=model_name)
                
        ax.set_xlabel('Training Size', fontsize=16, fontweight='bold')
        ax.set_ylabel('CATE MSE', fontsize=16, fontweight='bold')
        #ax.set_title(f'{metric} vs Training Size', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.legend(title='Models',
                title_fontsize=16,
                fontsize=14,
                loc='upper right',
                framealpha=0.9)
            
        #plt.tight_layout()
        self._save_plot("mse")

    def plot_MAE(self, results: Dict, train_sizes: list):
        _, ax = self._setup_plot()
        colors = self._get_colors(len(results) + 1)
        line_width = 2
        metrics = ["MAE"]

        for idx, (model_name, model_results) in enumerate(results.items()):
            mean_values = {metric: [] for metric in metrics}
            
            for size in train_sizes:
                for metric in metrics:
                    mean_values[metric].append(model_results[size]['mean'][metric])

            means = np.array(mean_values[metric])
            
            # Plot line connecting points
            ax.plot(train_sizes, means,
                    color=colors[idx + 1],
                    linewidth=line_width)
            
            # Plot scatter points on top of line
            ax.scatter(train_sizes, means,
                    color=colors[idx + 1],
                    s=50,
                    zorder=5,
                    label=model_name)
                
        ax.set_xlabel('Training Size', fontsize=16, fontweight='bold')
        ax.set_ylabel(metric, fontsize=16, fontweight='bold')
        #ax.set_title(f'{metric} vs Training Size', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.legend(title='Models',
                title_fontsize=16,
                fontsize=14,
                loc='upper right',
                framealpha=0.9)
            
        #plt.tight_layout()
        self._save_plot("mae")

    def plot_cdf_errors(self, results: Dict, train_sizes: list):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        max_size = max(train_sizes)
        colors = self._get_colors(len(results))
        
        for idx, (model_name, model_results) in enumerate(results.items()):
            errors_y0 = np.abs(model_results[max_size]['mean']['residuals'][:, 0])
            errors_y1 = np.abs(model_results[max_size]['mean']['residuals'][:, 1])
            
            for ax, errors, title in zip([ax1, ax2], [errors_y0, errors_y1], ['Y0', 'Y1']):
                sorted_errors = np.sort(errors)
                cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                ax.plot(sorted_errors, cumulative, label=model_name, color=colors[idx])
                ax.set_xlabel('Absolute Error', fontweight='bold')
                ax.set_ylabel('Cumulative Probability', fontweight='bold')
                ax.set_title(f'CDF of Estimation Errors ({title})')
                ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self._save_plot('cdf_errors')

    def plot_roc_curve(self, results: Dict, true_outcomes: Dict, train_sizes: list):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        max_size = max(train_sizes)
        colors = self._get_colors(len(results))
        
        for idx, (model_name, model_results) in enumerate(results.items()):
            for ax, y_true, y_pred, title in zip([ax1, ax2], 
                                               [true_outcomes['Y0'], true_outcomes['Y1']], 
                                               [model_results[max_size]['mean']['outcome_estimates'][:, 0],
                                                model_results[max_size]['mean']['outcome_estimates'][:, 1]], 
                                               ['Y0', 'Y1']):
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', 
                       color=colors[idx])
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontweight='bold')
                ax.set_ylabel('True Positive Rate', fontweight='bold')
                ax.set_title(f'ROC Curve ({title})')
                ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self._save_plot('roc_curve')

    def plot_cate_analysis(self, results: Dict, true_outcomes: Dict, train_sizes: list):

        max_size = max(train_sizes)
        true_cate = true_outcomes['Y1'] - true_outcomes['Y0']
        
        _, ax = self._setup_plot()
        colors = self._get_colors(len(results) + 1)
        
        sns.kdeplot(data=true_cate, 
                   label='True CATE',
                   color=colors[0],
                   alpha=1,
                   linewidth=1,
                   facecolor=(colors[0][0], colors[0][1], colors[0][2], 0.2),
                   fill=True)
        
        for idx, (model_name, model_results) in enumerate(results.items(), 1):
            estimated_cate = (model_results[max_size]['mean']['cate'])
            sns.kdeplot(data=estimated_cate,
                       label=model_name,
                       color=colors[idx],
                       alpha=0.8,
                       linewidth=1,
                       facecolor=(colors[idx][0], colors[idx][1], colors[idx][2], 0.2),
                       fill=True)
        
        ax.set_xlabel('Conditional Average Treatment Effect (CATE)', 
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Density', fontsize=16, fontweight='bold')
        #ax.set_title('Distribution of True and Estimated CATE\nAcross Different Models', 
        #            fontsize=18, fontweight='bold', pad=20)
        
        ax.legend(title='Models', 
                 title_fontsize=16,
                 fontsize=14,
                 loc='upper right')
        
        self._save_plot('cate_distributions')
        
        print("\nSummary Statistics:")
        print(f"True CATE: μ={true_cate.mean():.3f}, σ={true_cate.std():.3f}")
        for model_name, model_results in results.items():
            estimated_cate = (model_results[max_size]['mean']['outcome_estimates'][:, 1] - 
                            model_results[max_size]['mean']['outcome_estimates'][:, 0])
            print(f"{model_name}: μ={estimated_cate.mean():.3f}, σ={estimated_cate.std():.3f}")

    def create_all_plots(self, results: Dict, train_sizes: list, true_outcomes: Dict):

        self.plot_runtime_comparison(results, train_sizes)
        self.plot_cdf_errors(results, train_sizes)
        self.plot_fairness(results, train_sizes)
        self.plot_MAE(results, train_sizes)
        self.plot_cate_mse(results, train_sizes)
        self.plot_cate_analysis(results, true_outcomes, train_sizes)

def plot_results(results: Dict, train_sizes: list, Y0_test: np.ndarray, 
                Y1_test: np.ndarray, D_test: np.ndarray, save_path: str):
    true_outcomes = {'Y0': Y0_test, 'Y1': Y1_test}
    visualizer = ModelVisualizer(save_path)
    visualizer.create_all_plots(results, train_sizes, true_outcomes)