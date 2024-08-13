import pandas as pd
import datetime
import os
from monitoring.log_reader import LogReader

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import DataDriftPreset

import prefect

def generate_flow_run_name():
    time_now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"{time_now}"

@prefect.flow(
    name="NY Taxi - Data Drift and Quality Report", flow_run_name=generate_flow_run_name
)
def drift_quality_report() -> None:
    log_reader = LogReader()
    df_current = log_reader.download_logs(last_n_minutes=30)

    df_train = pd.read_csv('./data/train.csv')
    df_val = pd.read_csv('./data/val.csv')
    df_ref = pd.concat([df_train, df_val])

    X_cols = [col for col in df_current.columns if col not in ['prediction', 'trip_id']]

    evidently_token = os.getenv("EVIDENTLY_TOKEN")
    evidently_project_id = os.getenv("EVIDENTLY_PROJECT_ID")

    ws = CloudWorkspace(
        token=evidently_token,
        url="https://app.evidently.cloud")

    project = ws.get_project(evidently_project_id)

    data_report = Report(
        metrics=[
            DataDriftPreset(stattest='psi', stattest_threshold='0.3'),
            DataQualityPreset(),
        ],
    )

    data_report.run(reference_data=df_ref[X_cols], current_data=df_current[X_cols])

    ws.add_report(project.id, data_report)



if __name__ == "__main__":
    drift_quality_report.serve(name="NY Taxi - Scheduled Report", cron="*/5 * * * *")
