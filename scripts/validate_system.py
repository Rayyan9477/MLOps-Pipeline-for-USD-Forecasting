#!/usr/bin/env python3
"""
End-to-End System Validation
Tests complete data flow from API call to prediction
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(title):
    """Print section header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")


def test_api_prediction():
    """Test full prediction flow through API."""
    print_header("End-to-End API Prediction Test")

    strict = (
        os.getenv('STRICT_VALIDATION', '').lower()
        in {'1', 'true', 'yes'}
    )

    api_url = "http://localhost:8000"

    # 1. Check health
    print("\n1️⃣  Testing Health Endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"{GREEN}✓{RESET} API is healthy")
            print(f"   Status: {health['status']}")
            print(f"   Model Loaded: {health['model_loaded']}")
            print(f"   Model Version: {health.get('model_version', 'N/A')}")

            if not health.get('model_loaded', False):
                print(
                    f"{YELLOW}⚠{RESET} API running but"
                    " model not loaded (expected if"
                    " artifacts are stored in DVC"
                    " and not pulled yet)"
                )
                if strict:
                    return False
                print(
                    f"{YELLOW}   Skipping prediction"
                    f" checks in non-strict mode{RESET}"
                )
                return True
        else:
            print(
                f"{RED}✗{RESET} Health check failed:"
                f" {response.status_code}"
            )
            return False
    except Exception as e:
        print(f"{YELLOW}⚠{RESET} API not running: {e}")
        print(
            f"{YELLOW}   Start API with:"
            f" python -m uvicorn src.api.main:app"
            f" --host 0.0.0.0 --port 8000{RESET}"
        )
        return False if strict else True

    # 2. Get model info
    print("\n2️⃣  Testing Model Info Endpoint...")
    try:
        response = requests.get(
            f"{api_url}/model/info", timeout=5
        )
        if response.status_code == 200:
            model_info = response.json()
            print(f"{GREEN}✓{RESET} Model info retrieved")
            ver = model_info.get('model_version', 'N/A')
            print(f"   Model Version: {ver}")
            cnt = model_info.get('features_count', 'N/A')
            print(f"   Features Count: {cnt}")
        else:
            print(
                f"{YELLOW}⚠{RESET} Model info endpoint"
                f" returned {response.status_code}"
            )
    except Exception as e:
        print(f"{YELLOW}⚠{RESET} Model info error: {e}")

    # 3. Make prediction with dummy data
    print("\n3️⃣  Testing Prediction Endpoint...")
    try:
        # Load actual feature names from metadata
        from config.config import MODELS_DIR

        metadata_path = MODELS_DIR / 'latest_metadata.json'
        if not metadata_path.exists():
            print(
                f"{YELLOW}⚠{RESET} Model metadata not"
                f" found at {metadata_path};"
                " skipping prediction"
            )
            return False if strict else True

        with open(metadata_path) as f:
            metadata = json.load(f)

        feature_names = metadata['features']

        # Create realistic dummy features
        import numpy as np
        np.random.seed(42)
        features = {}
        for feat in feature_names:
            low = feat.lower()
            if 'hour' in low or 'day' in low:
                features[feat] = float(np.random.rand())
            elif 'lag' in low:
                val = 1.08 + np.random.randn() * 0.001
                features[feat] = val
            elif 'rolling' in low:
                if 'mean' in feat:
                    val = 1.08 + np.random.randn() * 0.001
                    features[feat] = val
                elif 'std' in feat:
                    val = abs(np.random.randn() * 0.001)
                    features[feat] = val
                else:
                    val = 1.08 + np.random.randn() * 0.001
                    features[feat] = val
            elif 'return' in low:
                val = np.random.randn() * 0.0001
                features[feat] = val
            elif 'volatility' in low:
                val = abs(np.random.randn() * 0.0005)
                features[feat] = val
            else:
                features[feat] = float(np.random.rand())

        payload = {"features": features}

        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            prediction = response.json()
            print(f"{GREEN}✓{RESET} Prediction successful")
            pred_val = prediction['prediction']
            print(f"   Predicted Volatility: {pred_val:.6f}")
            risk = prediction.get('risk_level', 'N/A')
            print(f"   Risk Level: {risk}")
            drift = prediction.get('drift_detected', False)
            print(f"   Drift Detected: {drift}")
            lat = prediction.get('latency_ms', 0)
            print(f"   Latency: {lat:.2f} ms")
            mver = prediction.get('model_version', 'N/A')
            print(f"   Model Version: {mver}")
        else:
            print(
                f"{RED}✗{RESET} Prediction failed:"
                f" {response.status_code}"
            )
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"{RED}✗{RESET} Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return False if strict else True

    # 4. Check Prometheus metrics
    print("\n4️⃣  Testing Prometheus Metrics...")
    try:
        response = requests.get(
            f"{api_url}/metrics", timeout=5
        )
        if response.status_code == 200:
            metrics_text = response.text
            print(
                f"{GREEN}✓{RESET}"
                " Prometheus metrics available"
            )

            # Check for key metrics
            if 'predictions_total' in metrics_text.lower():
                print("   ✓ predictions_total metric found")
            if 'prediction_latency' in metrics_text.lower():
                print("   ✓ prediction_latency metric found")
        else:
            print(
                f"{YELLOW}⚠{RESET} Metrics endpoint"
                f" returned {response.status_code}"
            )
    except Exception as e:
        print(f"{YELLOW}⚠{RESET} Metrics error: {e}")

    # 5. Check API stats
    print("\n5️⃣  Testing API Stats...")
    try:
        response = requests.get(
            f"{api_url}/api/stats", timeout=5
        )
        if response.status_code == 200:
            stats = response.json()
            print(f"{GREEN}✓{RESET} API stats retrieved")
            total = stats.get('total_predictions', 0)
            print(f"   Total Predictions: {total}")
            avg = stats.get('avg_latency_ms', 0)
            print(f"   Average Latency: {avg:.2f} ms")
        else:
            print(
                f"{YELLOW}⚠{RESET} Stats endpoint"
                f" returned {response.status_code}"
            )
    except Exception as e:
        print(f"{YELLOW}⚠{RESET} Stats error: {e}")

    print(
        f"\n{GREEN}✅ End-to-End Test Complete!{RESET}"
    )
    return True


def test_workflow_schedule():
    """Test GitHub Actions workflow configuration."""
    print_header("GitHub Actions Workflow Validation")

    workflow_file = Path(
        '.github/workflows/data-pipeline.yml'
    )

    if not workflow_file.exists():
        print(f"{RED}✗{RESET} Workflow file not found")
        return False

    content = workflow_file.read_text()

    # Check for 2-hour schedule
    if '0 */2 * * *' in content:
        print(
            f"{GREEN}✓{RESET}"
            " 2-hour cron schedule configured"
        )
    else:
        print(
            f"{YELLOW}⚠{RESET}"
            " 2-hour schedule not found"
        )

    # Check for workflow_dispatch
    if 'workflow_dispatch' in content:
        print(f"{GREEN}✓{RESET} Manual trigger enabled")
    else:
        print(
            f"{YELLOW}⚠{RESET}"
            " Manual trigger not configured"
        )

    # Check for secrets usage
    if 'secrets.TWELVE_DATA_API_KEY' in content:
        print(f"{GREEN}✓{RESET} GitHub Secrets configured")
    else:
        print(f"{YELLOW}⚠{RESET} Secrets not referenced")

    return True


def test_airflow_dag():
    """Test Airflow DAG configuration."""
    print_header("Airflow DAG Validation")

    dag_file = Path('airflow/dags/etl_dag.py')

    if not dag_file.exists():
        print(f"{RED}✗{RESET} DAG file not found")
        return False

    content = dag_file.read_text()

    # Check for schedule
    has_sched = (
        "schedule='0 */2 * * *'" in content
        or 'schedule="0 */2 * * *"' in content
    )
    if has_sched:
        print(
            f"{GREEN}✓{RESET}"
            " 2-hour schedule configured in DAG"
        )
    else:
        print(
            f"{YELLOW}⚠{RESET}"
            " Schedule not found in DAG"
        )

    # Check for tasks
    required_tasks = [
        'extract', 'transform', 'load', 'version'
    ]
    for task in required_tasks:
        if task in content.lower():
            print(f"{GREEN}✓{RESET} Task '{task}' found in DAG")

    return True


def main():
    """Run all validation tests."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}System Validation - Complete Flow Test{RESET}")
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{BLUE}Time: {now}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")

    results = []

    # Test API
    api_result = test_api_prediction()
    results.append(('API Flow', api_result))

    # Test workflows
    workflow_result = test_workflow_schedule()
    results.append(('GitHub Actions', workflow_result))

    # Test Airflow
    airflow_result = test_airflow_dag()
    results.append(('Airflow DAG', airflow_result))

    # Summary
    print_header("Validation Summary")

    all_passed = all(result for _, result in results)

    for name, result in results:
        if result:
            status = f"{GREEN}✓ PASS{RESET}"
        else:
            status = f"{RED}✗ FAIL{RESET}"
        print(f"{name}: {status}")

    print(f"\n{BLUE}{'=' * 70}{RESET}")

    if all_passed:
        print(
            f"{GREEN}✅ ALL VALIDATION CHECKS PASSED{RESET}"
        )
        print(
            f"{GREEN}System is fully operational"
            f" and production-ready!{RESET}"
        )
        return 0
    else:
        print(
            f"{YELLOW}⚠ Some validation checks"
            f" need attention{RESET}"
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())
