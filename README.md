# ITU BDS MLOPS'25 - Project

This repository contains our implementation of a fully reproducible end-to-end MLOps pipeline developed for the ITU BDS MLOps 2025 course. The goal was to transform the original exploratory notebook into a modular, automated ML system using DVC for data management, MLflow for experiment tracking, and GitHub Actions + Dagger for CI/CD automation.

## Project Architecture

### 1. Repository Structure (Files tracked by Git)

```text
itu-sdse-project/
│
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml     # CI/CD Orchestration: Executes Dagger, then uploads the generated artifacts.
│
├── data/
│   └── raw_data.csv.dvc        # DVC Pointer: The only data-related file tracked by Git.
│
├── src/                        # Core Python Pipeline Logic (Input)
│   ├── data_prep.py            # Phase 1: Executes 'dvc pull' and performs data cleaning.
│   ├── train.py                # Phase 2: Training, MLflow run logging, and local model saving.
│   └── deploy.py               # Phase 3: Model selection, registration to the 'mlruns' directory.
│
├── main.go                     # Orchestrator (Dagger): Defines the pipeline steps and exports the generated folders.
├── requirements.txt            # Dependencies: Defines the Python environment for the Dagger container.
└── README.md
