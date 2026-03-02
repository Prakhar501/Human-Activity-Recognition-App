"""
Comprehensive Evaluation Metrics Module

Tracks and stores detailed evaluation metrics including:
- Model accuracy and loss
- Confusion matrices
- Precision, Recall, F1-Score
- Comparison between local and federated models
- ROC curves and other metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from datetime import datetime
import json
import os
from pathlib import Path


class EvaluationMetrics:
    """Comprehensive evaluation metrics tracker"""
    
    def __init__(self, model_name="model", activity_labels=None):
        """
        Initialize evaluation metrics tracker
        
        Args:
            model_name: Name of the model (e.g., "federated", "local", "centralized")
            activity_labels: List of activity class names
        """
        self.model_name = model_name
        self.activity_labels = activity_labels or ['Walking', 'Walking_Upstairs', 'Walking_Downstairs', 'Sitting']
        self.num_classes = len(self.activity_labels)
        
        # Metrics storage
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics_dict = {}
        self.confusion_matrix = None
        self.per_class_metrics = {}
        
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for advanced metrics)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        # Overall metrics
        self.metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred, 
                                                 labels=range(self.num_classes))
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Store per-class metrics
        for i, label in enumerate(self.activity_labels):
            self.per_class_metrics[label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(np.sum(y_true == i))
            }
        
        # Macro averages
        self.metrics_dict['precision_macro'] = float(np.mean(precision))
        self.metrics_dict['recall_macro'] = float(np.mean(recall))
        self.metrics_dict['f1_macro'] = float(np.mean(f1))
        
        # Weighted averages
        self.metrics_dict['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        self.metrics_dict['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        self.metrics_dict['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        return self.metrics_dict
    
    def get_classification_report(self):
        """Get detailed classification report"""
        if self.y_true is None:
            return "No evaluation data available"
        
        return classification_report(
            self.y_true, 
            self.y_pred,
            target_names=self.activity_labels,
            digits=4
        )
    
    def get_confusion_matrix_dict(self):
        """Return confusion matrix as dictionary"""
        if self.confusion_matrix is None:
            return {}
        
        cm_dict = {}
        for i, true_label in enumerate(self.activity_labels):
            cm_dict[true_label] = {}
            for j, pred_label in enumerate(self.activity_labels):
                cm_dict[true_label][pred_label] = int(self.confusion_matrix[i, j])
        
        return cm_dict
    
    def to_dict(self):
        """Convert all metrics to dictionary"""
        return {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': self.metrics_dict,
            'per_class_metrics': self.per_class_metrics,
            'confusion_matrix': self.get_confusion_matrix_dict(),
            'activity_labels': self.activity_labels
        }
    
    def to_json(self, filepath):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Metrics saved to {filepath}")
    
    def create_comparison_table(self, other_metrics):
        """
        Create comparison table between this model and another
        
        Args:
            other_metrics: Another EvaluationMetrics object
            
        Returns:
            pandas DataFrame with comparison
        """
        comparison_data = {
            'Metric': [
                'Accuracy',
                'Precision (Macro)',
                'Recall (Macro)',
                'F1-Score (Macro)',
                'Precision (Weighted)',
                'Recall (Weighted)',
                'F1-Score (Weighted)'
            ],
            self.model_name: [
                f"{self.metrics_dict.get('accuracy', 0):.4f}",
                f"{self.metrics_dict.get('precision_macro', 0):.4f}",
                f"{self.metrics_dict.get('recall_macro', 0):.4f}",
                f"{self.metrics_dict.get('f1_macro', 0):.4f}",
                f"{self.metrics_dict.get('precision_weighted', 0):.4f}",
                f"{self.metrics_dict.get('recall_weighted', 0):.4f}",
                f"{self.metrics_dict.get('f1_weighted', 0):.4f}",
            ]
        }
        
        if other_metrics:
            comparison_data[other_metrics.model_name] = [
                f"{other_metrics.metrics_dict.get('accuracy', 0):.4f}",
                f"{other_metrics.metrics_dict.get('precision_macro', 0):.4f}",
                f"{other_metrics.metrics_dict.get('recall_macro', 0):.4f}",
                f"{other_metrics.metrics_dict.get('f1_macro', 0):.4f}",
                f"{other_metrics.metrics_dict.get('precision_weighted', 0):.4f}",
                f"{other_metrics.metrics_dict.get('recall_weighted', 0):.4f}",
                f"{other_metrics.metrics_dict.get('f1_weighted', 0):.4f}",
            ]
        
        return pd.DataFrame(comparison_data)
    
    def plot_confusion_matrix(self, filepath=None, figsize=(10, 8)):
        """Plot confusion matrix heatmap"""
        if self.confusion_matrix is None:
            print("No confusion matrix available")
            return
        
        plt.figure(figsize=figsize)
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.activity_labels))
        plt.xticks(tick_marks, self.activity_labels, rotation=45)
        plt.yticks(tick_marks, self.activity_labels)
        
        # Add text annotations
        for i in range(len(self.activity_labels)):
            for j in range(len(self.activity_labels)):
                plt.text(j, i, str(self.confusion_matrix[i, j]),
                        horizontalalignment="center",
                        color="white" if self.confusion_matrix[i, j] > self.confusion_matrix.max() / 2 else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix plot saved to {filepath}")
        
        return plt
    
    def plot_per_class_metrics(self, filepath=None, figsize=(12, 6)):
        """Plot per-class metrics comparison"""
        if not self.per_class_metrics:
            print("No per-class metrics available")
            return
        
        labels = list(self.per_class_metrics.keys())
        precision = [self.per_class_metrics[l]['precision'] for l in labels]
        recall = [self.per_class_metrics[l]['recall'] for l in labels]
        f1 = [self.per_class_metrics[l]['f1_score'] for l in labels]
        
        x = np.arange(len(labels))
        width = 0.25
        
        plt.figure(figsize=figsize)
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Activity Class')
        plt.ylabel('Score')
        plt.title(f'Per-Class Metrics - {self.model_name}')
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        plt.ylim([0, 1.05])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Per-class metrics plot saved to {filepath}")
        
        return plt


class MetricsReporter:
    """Generate comprehensive metrics reports"""
    
    def __init__(self, output_dir='results'):
        """Initialize reporter"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models_metrics = {}
    
    def add_model_metrics(self, model_name, metrics_obj):
        """Add model metrics"""
        self.models_metrics[model_name] = metrics_obj
    
    def generate_comparison_report(self):
        """Generate comparison report of all models"""
        if len(self.models_metrics) < 1:
            print("No models to compare")
            return None
        
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE EVALUATION METRICS REPORT")
        report.append("=" * 100)
        report.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall comparison table
        report.append("\n" + "=" * 100)
        report.append("1. OVERALL METRICS COMPARISON")
        report.append("=" * 100 + "\n")
        
        comparison_data = []
        for model_name, metrics in self.models_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics.metrics_dict.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.metrics_dict.get('precision_macro', 0):.4f}",
                'Recall': f"{metrics.metrics_dict.get('recall_macro', 0):.4f}",
                'F1-Score': f"{metrics.metrics_dict.get('f1_macro', 0):.4f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        report.append(comparison_df.to_string(index=False))
        report.append("\n")
        
        # Per-model detailed metrics
        report.append("\n" + "=" * 100)
        report.append("2. DETAILED METRICS PER MODEL")
        report.append("=" * 100 + "\n")
        
        for model_name, metrics in self.models_metrics.items():
            report.append(f"\n### {model_name.upper()} ###\n")
            report.append(metrics.get_classification_report())
            
            # Confusion matrix
            report.append("\nConfusion Matrix:\n")
            cm_dict = metrics.get_confusion_matrix_dict()
            cm_df = pd.DataFrame(cm_dict).T
            report.append(cm_df.to_string())
            report.append("\n")
        
        return "\n".join(report)
    
    def save_report(self, filename='evaluation_report.txt'):
        """Save comprehensive report to file"""
        report = self.generate_comparison_report()
        if report:
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"✅ Report saved to {filepath}")
            return filepath
    
    def save_all_metrics(self, json_filename='all_metrics.json'):
        """Save all metrics to JSON"""
        all_metrics = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, metrics in self.models_metrics.items():
            all_metrics['models'][model_name] = metrics.to_dict()
        
        filepath = self.output_dir / json_filename
        with open(filepath, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"✅ All metrics saved to {filepath}")
        return filepath
    
    def create_comparison_plots(self):
        """Create comparison plots for all models"""
        if len(self.models_metrics) < 1:
            print("No models to plot")
            return
        
        # Create comparison figure
        models = list(self.models_metrics.keys())
        accuracies = [self.models_metrics[m].metrics_dict.get('accuracy', 0) for m in models]
        f1_scores = [self.models_metrics[m].metrics_dict.get('f1_macro', 0) for m in models]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        axes[0].bar(models, accuracies, alpha=0.7, color='steelblue')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylim([0, 1])
        for i, acc in enumerate(accuracies):
            axes[0].text(i, acc + 0.02, f'{acc:.4f}', ha='center')
        
        # F1-Score comparison
        axes[1].bar(models, f1_scores, alpha=0.7, color='seagreen')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('F1-Score Comparison (Macro)')
        axes[1].set_ylim([0, 1])
        for i, f1 in enumerate(f1_scores):
            axes[1].text(i, f1 + 0.02, f'{f1:.4f}', ha='center')
        
        plt.tight_layout()
        filepath = self.output_dir / 'models_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {filepath}")
        
        plt.close()
    
    def print_summary(self):
        """Print summary to console"""
        print("\n" + "=" * 100)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 100)
        
        for model_name, metrics in self.models_metrics.items():
            print(f"\n📊 {model_name.upper()}")
            print(f"   Accuracy:      {metrics.metrics_dict.get('accuracy', 0):.4f}")
            print(f"   Precision:     {metrics.metrics_dict.get('precision_macro', 0):.4f}")
            print(f"   Recall:        {metrics.metrics_dict.get('recall_macro', 0):.4f}")
            print(f"   F1-Score:      {metrics.metrics_dict.get('f1_macro', 0):.4f}")


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    y_pred_federated = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 3, 3, 0, 1, 2, 2])
    y_pred_local = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    
    # Create metrics objects
    federated_metrics = EvaluationMetrics('Federated Model')
    federated_metrics.evaluate(y_true, y_pred_federated)
    
    local_metrics = EvaluationMetrics('Local Model')
    local_metrics.evaluate(y_true, y_pred_local)
    
    # Create reporter
    reporter = MetricsReporter()
    reporter.add_model_metrics('federated', federated_metrics)
    reporter.add_model_metrics('local', local_metrics)
    
    # Generate and save reports
    print(reporter.generate_comparison_report())
    reporter.save_report()
    reporter.save_all_metrics()
    reporter.create_comparison_plots()
    reporter.print_summary()
    
    print("\n✅ All metrics generated successfully!")
