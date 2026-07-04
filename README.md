
# AI-Based Real-Time Aerial Threat Classification System

## Project Overview
This project builds an AI-powered radar threat classification system for identifying aerial objects including:

- Ballistic Missile
- Hypersonic Missile
- Cruise Missile
- Fighter Jet
- Commercial Aircraft
- Private/Civilian Jet
- UAV/Drone

## Dataset Features Used

- speed_kmh
- altitude_m
- direction_deg
- rcs_sqm
- acceleration_g

## Features Removed (Leakage Prevention)

The following columns were intentionally excluded:

- object_id
- timestamp
- trajectory_type
- iff_response
- threat_level

Reason:
These fields directly encode or strongly reveal target information and would create target leakage.

## Models

Baseline:
- Random Forest

Main model:
- XGBoost
- 5-fold Stratified Cross Validation
- RandomizedSearchCV tuning

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Feature Importance

## Results Folder

Contains:

- confusion_matrix.png
- feature_importance.png
- classification_report.txt


## Limitations

- Dataset is synthetic and relatively small (200 samples)
- Real radar environments contain sensor noise and changing conditions
- Performance on operational radar systems may differ significantly....
- Future work could include time-series tracking, Kalman filtering, and real radar feeds...
