"""
Airflow DAG for Plant Disease Detection ML Pipeline
Orchestrates data preprocessing, feature engineering, model training, and evaluation
"""

from datetime import timedelta
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add src directory to Python path for imports
sys.path.insert(0, "/opt/airflow/dags/src")

# Default arguments for the DAG
default_args = {
    "owner": "plant-disease-ml-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "plant_disease_ml_pipeline",
    default_args=default_args,
    description="Complete ML pipeline for plant disease detection",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=["ml", "plant-disease", "classification"],
)


def preprocess_data_task(**context):
    """Data preprocessing task: load, clean, and split data"""
    try:
        # Import here to avoid issues with path
        from data_preprocessing import preprocess_data

        print("Starting data preprocessing...")
        X_train, X_test, y_train, y_test = preprocess_data()

        print("Data preprocessing completed:")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}")

        # Store metadata in XCom for next tasks
        context["task_instance"].xcom_push(
            key="data_info",
            value={
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "features": X_train.shape[1],
            },
        )

        return "Data preprocessing completed successfully"

    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise


def train_model_task(**context):
    """Model training task: train and select best model"""
    try:
        from model_training import train_model

        print("Starting model training...")

        # Get data info from previous task
        data_info = context["task_instance"].xcom_pull(
            task_ids="preprocess_data", key="data_info"
        )
        print(
            f"Training on {data_info['train_samples']} samples with {data_info['features']} features"
        )

        model, model_name = train_model()

        print("Model training completed:")
        print(f"Best model: {model_name}")

        # Store model info in XCom
        context["task_instance"].xcom_push(
            key="model_info", value={"model_name": model_name, "model_saved": True}
        )

        return f"Model training completed - Best model: {model_name}"

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise


def evaluate_model_task(**context):
    """Model evaluation task: test model performance"""
    try:
        from evaluation import evaluate_saved_model

        print("Starting model evaluation...")

        # Get model info from previous task
        model_info = context["task_instance"].xcom_pull(
            task_ids="train_model", key="model_info"
        )
        print(f"Evaluating {model_info['model_name']} model")

        metrics = evaluate_saved_model()

        print("Model evaluation completed:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")

        # Store final results in XCom
        context["task_instance"].xcom_push(
            key="final_metrics",
            value={
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
            },
        )

        return f"Model evaluation completed - Accuracy: {metrics['accuracy']:.4f}"

    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        raise


def pipeline_summary_task(**context):
    """Final task: summarize pipeline results"""
    try:
        # Get results from all previous tasks
        data_info = context["task_instance"].xcom_pull(
            task_ids="preprocess_data", key="data_info"
        )
        model_info = context["task_instance"].xcom_pull(
            task_ids="train_model", key="model_info"
        )
        metrics = context["task_instance"].xcom_pull(
            task_ids="evaluate_model", key="final_metrics"
        )

        print("\n" + "=" * 60)
        print("PLANT DISEASE DETECTION ML PIPELINE SUMMARY")
        print("=" * 60)
        print(
            f"Dataset: {data_info['train_samples']} training + {data_info['test_samples']} test samples"
        )
        print(f"Features: {data_info['features']}")
        print(f"Best Model: {model_info['model_name']}")
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
        print(f"Final Precision: {metrics['precision']:.4f}")
        print(f"Final Recall: {metrics['recall']:.4f}")
        print(f"Final F1-Score: {metrics['f1_score']:.4f}")
        print("=" * 60)
        print("Pipeline completed successfully!")

        return "ML Pipeline completed successfully"

    except Exception as e:
        print(f"Error in pipeline summary: {str(e)}")
        raise


# Define tasks
preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data_task,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model_task,
    provide_context=True,
    dag=dag,
)

summary_task = PythonOperator(
    task_id="pipeline_summary",
    python_callable=pipeline_summary_task,
    provide_context=True,
    dag=dag,
)

# Define task dependencies (this creates the workflow)
preprocess_task >> train_task >> evaluate_task >> summary_task
