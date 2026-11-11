"""Streamlit dashboard for monitoring fraud detection system."""

import os
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8001").rstrip("/")
AUTO_REFRESH = os.getenv("DASHBOARD_AUTO_REFRESH", "true").lower() == "true"


def fetch_metrics():
    """Fetch metrics from the API, returning None when unavailable."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None


def main():
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        page_icon="🚨",
        layout="wide"
    )

    st.title("🚨 Real-Time Fraud Detection Dashboard")
    st.markdown("---")

    if "history" not in st.session_state:
        st.session_state.history = []

    metrics = fetch_metrics()
    ml_metrics = (metrics or {}).get("ml", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", ml_metrics.get("total_predictions", 0))
    col2.metric("Fraud Detected", ml_metrics.get("fraud_count", 0))
    col3.metric("Fraud Rate", f"{ml_metrics.get('fraud_rate', 0):.1f}%")
    if metrics is None:
        col4.error("❌ API: Connection Error")
    elif metrics.get("model_loaded"):
        col4.success("✅ API: Healthy")
    else:
        col4.warning("⚠️ API: Model Not Loaded")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Predictions",
        "⚠️ Alerts",
        "📈 Analytics"
    ])

    with tab1:
        st.subheader("System Status")
        if metrics is None:
            st.error(f"Could not reach the API at {API_URL}")
        else:
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.write("**Model**")
                st.json({
                    "type": metrics.get("model_type"),
                    "loaded": metrics.get("model_loaded"),
                    "features": metrics.get("feature_count")
                })
            with status_col2:
                st.write("**System**")
                st.json(metrics.get("system", {}))

        st.markdown("### Recent Activity")
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history))
        else:
            st.info("No predictions yet. Submit one from the Predictions tab.")

    with tab2:
        st.subheader("Make a Prediction")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                transaction_id = st.text_input("Transaction ID", "TX001")
                amount = st.number_input("Amount ($)", min_value=0.0, value=450000.0)
                hour = st.selectbox("Hour of Day", range(0, 24), index=14)
                location = st.text_input("Location", "Columbo")

            with col2:
                is_new_device = st.selectbox("New Device?", [0, 1], index=1)
                is_foreign_location = st.selectbox("Foreign Location?", [0, 1], index=1)
                transactions_last_hour = st.number_input(
                    "Transactions Last Hour",
                    min_value=0,
                    value=8
                )
                previous_transaction_amount = st.number_input(
                    "Previous Amount ($)",
                    min_value=0.0,
                    value=2500.0
                )

            submitted = st.form_submit_button("Predict")

            if submitted:
                transaction_data = {
                    "transaction_id": transaction_id,
                    "amount": amount,
                    "hour": hour,
                    "is_new_device": is_new_device,
                    "is_foreign_location": is_foreign_location,
                    "transactions_last_hour": transactions_last_hour,
                    "previous_transaction_amount": previous_transaction_amount,
                    "location": location
                }

                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=transaction_data,
                        timeout=5
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.session_state.history.insert(0, {
                            "transaction_id": result["transaction_id"],
                            "amount": amount,
                            "prediction": result["prediction"],
                            "fraud_probability": round(result["fraud_probability"], 4),
                            "time": datetime.now().strftime("%H:%M:%S")
                        })
                        st.session_state.history = st.session_state.history[:100]

                        result_col1, result_col2, result_col3 = st.columns(3)
                        result_col1.metric("Prediction", result["prediction"])
                        result_col2.metric(
                            "Fraud Probability",
                            f"{result['fraud_probability']:.2%}"
                        )
                        status = "🚨 FRAUD" if result["is_fraud"] == 1 else "✅ LEGITIMATE"
                        result_col3.metric("Status", status)

                        if result["is_fraud"] == 1:
                            st.error("⚠️ Suspicious transaction detected!")
                        else:
                            st.success("Transaction appears legitimate")
                    else:
                        st.error(f"Prediction failed: {response.text}")
                except requests.RequestException as exc:
                    st.error(f"Error: {exc}")

    with tab3:
        st.subheader("Fraud Alerts")
        frauds = [
            item for item in st.session_state.history
            if item["prediction"] == "FRAUD"
        ]
        if frauds:
            st.dataframe(pd.DataFrame(frauds))
        else:
            st.info("No fraud alerts yet.")

    with tab4:
        st.subheader("Model Performance Analytics")

        if metrics is None:
            st.info("Connect to the API to view live analytics.")
        else:
            st.metric("Model Type", metrics.get("model_type") or "N/A")

            history_df = pd.DataFrame(st.session_state.history)
            if not history_df.empty:
                st.markdown("### Prediction Distribution")
                counts = history_df["prediction"].value_counts()
                st.bar_chart(counts)

                st.markdown("### Fraud Probability Over Time")
                st.line_chart(
                    history_df.reset_index(drop=True)["fraud_probability"]
                )
            else:
                st.info("Charts will appear after predictions are made.")

    if AUTO_REFRESH:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
