"""
Shared session state initialisation for the Streamlit app.
Call init_session_state() at the top of every page to ensure all keys exist.
"""

import streamlit as st
from src.monitoring.alerts import AlertManager


def init_session_state():
    """
    Initialise all session state keys with default values.
    Safe to call multiple times — only sets keys that don't already exist.
    """
    defaults = {
        # Prediction history accumulated during the session
        "prediction_history": [],   # list of dicts from model_service.predict_single()

        # AlertManager instance — accumulates alerts across predictions
        "alert_manager": None,      # initialised lazily below (avoid pickling issues)

        # Fitted DriftDetector (set by Monitoring page after fit() on reference data)
        "drift_detector": None,

        # Log lines captured from pipeline/training runs
        "pipeline_log": [],

        # Counter used to invalidate @st.cache_resource model cache on reload
        "model_reload_count": 0,

        # Stores the most recent single-prediction result dict
        "last_prediction": None,

        # Stores raw data DataFrame from Data Pipeline page (session scope)
        "raw_data": None,

        # Stores transformed data DataFrame from Data Pipeline page
        "processed_data": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            if key == "alert_manager":
                st.session_state[key] = AlertManager()
            else:
                st.session_state[key] = default_value
