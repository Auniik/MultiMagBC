"""
Evaluation and Visualization Script for MMNet
Generates plots and tables from saved training results
"""

import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import seaborn as sns

class MMEvaluator:
    def __init__(self, results_dir='./output/results'):
        self.results_dir = results_dir
        self.json_files = []
        self.csv_summary = None
        self.load_results()
    
    def load_results(self):
        """Load all JSON and CSV results files"""
        # Load JSON files
        json_pattern = os.path.join(self.results_dir, 'fold_*.json')
        self.json_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        # Load CSV summary
        csv_path = os.path.join(self.results_dir, 'results_summary.csv')
        if os.path.exists(csv_path):
            self.csv_summary = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.json_files)} JSON files and CSV summary")
    
    def generate_learning_curves(self, fold_idx=None):
        """Generate learning curves for specified fold or all folds"""
        if fold_idx is not None:
            files = [f'fold_{fold_idx}.json']
        else:
            files = self.json_files
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, json_file in enumerate(files):
            if i >= 4:  # Limit to 4 subplots
                break
                
            with open(os.path.join(self.results_dir, json_file), 'r') as f:
                data = json.load(f)
            
            history = data['training_history']
            epochs = range(1, len(history['train_losses']) + 1)
            
            # Loss curves
            axes[i//2, i%2].plot(epochs, history['train_losses'], 'b-', label='Train Loss', alpha=0.7)
            axes[i//2, i%2].plot(epochs, history['val_losses'], 'r-', label='Val Loss', alpha=0.7)
            axes[i//2, i%2].set_title(f'Fold {i} - Learning Curves')
            axes[i//2, i%2].set_xlabel('Epoch')
            axes[i//2, i%2].set_ylabel('Loss')
            axes[i//2, i%2].legend()
            axes[i//2, i%2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, 'learning_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Learning curves saved: {output_path}")
    
    def generate_roc_curves(self, fold_idx=None):
        """Generate ROC curves for specified fold or all folds"""
        if fold_idx is not None:
            files = [f'fold_{fold_idx}.json']
        else:
            files = self.json_files
        
        plt.figure(figsize=(10, 8))
        
        for json_file in files:
            with open(os.path.join(self.results_dir, json_file), 'r') as f:
                data = json.load(f)
            
            roc_data = data['roc_data']
            fold_num = data['fold']
            
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'Fold {fold_num} (AUC = {data["test_auc"]:.3f})', 
                    linewidth=2)
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.results_dir, 'roc_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 ROC curves saved: {output_path}")
    
    def generate_performance_table(self):
        """Generate comprehensive performance table"""
        if self.csv_summary is None:
            print("❌ No CSV summary found")
            return
        
        # Create formatted table
        table = self.csv_summary.round(4)
        
        # Add summary row
        summary = table.mean().round(4)
        summary['fold'] = 'Mean'
        summary_row = pd.DataFrame([summary])
        
        # Add std row
        std_row = table.std().round(4)
        std_row['fold'] = 'Std'
        std_row = pd.DataFrame([std_row])
        
        # Combine
        full_table = pd.concat([table, summary_row, std_row], ignore_index=True)
        
        # Save to CSV
        output_path = os.path.join(self.results_dir, 'performance_table.csv')
        full_table.to_csv(output_path, index=False)
        
        # Print formatted table
        print("\n📊 Performance Summary Table:")
        print("=" * 80)
        print(full_table.to_string(index=False))
        print("=" * 80)
        
        return full_table
    
    def generate_magnification_analysis(self):
        """Analyze and visualize magnification importance across folds"""
        magnitudes = {'40x': [], '100x': [], '200x': [], '400x': []}
        
        for json_file in self.json_files:
            with open(os.path.join(self.results_dir, json_file), 'r') as f:
                data = json.load(f)
            
            importance = data['magnification_importance']
            for mag, value in importance.items():
                magnitudes[mag].append(value)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot
        mag_data = [magnitudes[mag] for mag in ['40x', '100x', '200x', '400x']]
        bp = ax.boxplot(mag_data, labels=['40x', '100x', '200x', '400x'], patch_artist=True)
        
        # Color boxes
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Magnification Importance Across Folds')
        ax.set_ylabel('Importance Score')
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.results_dir, 'magnification_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        mag_stats = pd.DataFrame({
            'magnification': ['40x', '100x', '200x', '400x'],
            'mean': [np.mean(magnitudes[mag]) for mag in ['40x', '100x', '200x', '400x']],
            'std': [np.std(magnitudes[mag]) for mag in ['40x', '100x', '200x', '400x']],
            'min': [np.min(magnitudes[mag]) for mag in ['40x', '100x', '200x', '400x']],
            'max': [np.max(magnitudes[mag]) for mag in ['40x', '100x', '200x', '400x']]
        })
        
        mag_stats.to_csv(os.path.join(self.results_dir, 'magnification_stats.csv'), index=False)
        print(f"💾 Magnification analysis saved")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.csv_summary is None:
            return
        
        metrics = ['accuracy', 'balanced_accuracy', 'f1', 'auc', 'precision', 'recall']
        
        # Calculate summary statistics
        summary = {
            'metric': metrics,
            'mean': [self.csv_summary[metric].mean() for metric in metrics],
            'std': [self.csv_summary[metric].std() for metric in metrics],
            'min': [self.csv_summary[metric].min() for metric in metrics],
            'max': [self.csv_summary[metric].max() for metric in metrics],
            '95_ci_lower': [self.csv_summary[metric].mean() - 1.96 * self.csv_summary[metric].std() / np.sqrt(len(self.csv_summary)) for metric in metrics],
            '95_ci_upper': [self.csv_summary[metric].mean() + 1.96 * self.csv_summary[metric].std() / np.sqrt(len(self.csv_summary)) for metric in metrics]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.round(4)
        
        # Save summary
        summary_df.to_csv(os.path.join(self.results_dir, 'summary_report.csv'), index=False)
        
        print("\n📋 Final Summary Report:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
        
        return summary_df
    
    def run_all_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting comprehensive analysis...")
        
        if not self.json_files:
            print("❌ No results found. Please run training first.")
            return
        
        # Generate all visualizations
        self.generate_learning_curves()
        self.generate_roc_curves()
        self.generate_performance_table()
        self.generate_magnification_analysis()
        self.generate_summary_report()
        
        print("\n✅ Analysis complete! Check the results directory for:")
        print("   - learning_curves.png")
        print("   - roc_curves.png") 
        print("   - magnification_importance.png")
        print("   - performance_table.csv")
        print("   - summary_report.csv")
        print(f"   📁 Location: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='MMNet Evaluation and Visualization')
    parser.add_argument('--results-dir', default='./output/results', help='Directory containing results')
    parser.add_argument('--fold', type=int, help='Specific fold to analyze (optional)')
    parser.add_argument('--learning-curves', action='store_true', help='Generate learning curves')
    parser.add_argument('--roc-curves', action='store_true', help='Generate ROC curves')
    parser.add_argument('--table', action='store_true', help='Generate performance table')
    parser.add_argument('--magnitude', action='store_true', help='Generate magnification analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses (default)')
    
    args = parser.parse_args()
    
    evaluator = MMEvaluator(args.results_dir)
    
    if args.all or not any([args.learning_curves, args.roc_curves, args.table, args.magnitude]):
        evaluator.run_all_analysis()
    else:
        if args.learning_curves:
            evaluator.generate_learning_curves(args.fold)
        if args.roc_curves:
            evaluator.generate_roc_curves(args.fold)
        if args.table:
            evaluator.generate_performance_table()
        if args.magnitude:
            evaluator.generate_magnification_analysis()


if __name__ == "__main__":
    main()