# Probability of Default Model: End-to-End ML on Snowflake

## Demo Design Document

A notebook-based demonstration showing how a BNPL (Buy Now, Pay Later) fintech can migrate their Probability of Default (PD) ML pipeline from AWS SageMaker to Snowflake, achieving operational simplicity, cost reduction, and competitive inference performance.

**Data scale**: 500k training applications (20 features), 600k inference records, DEV/STAGING/PROD environments.

---

## Executive Summary

**Problem**: A BNPL fintech currently runs their PD model pipeline across multiple AWS services -- SageMaker for training/serving, S3 for staging, Terraform for infrastructure, CloudWatch + EventBridge for monitoring and alerting. Training data already lives in Snowflake, creating unnecessary data movement, operational complexity, and cost overhead across disconnected platforms.

**Solution**: Consolidate the entire ML lifecycle into Snowflake -- Feature Store, Notebooks, Model Registry, SPCS-based real-time serving, and ML Observability -- eliminating data movement, reducing infrastructure to manage, and providing unified governance with built-in DEV/STAGING/PROD environment separation via database isolation and RBAC.

---

## Architecture

### Current State (SageMaker)

```
Snowflake (Data) --> S3 (Staging) --> SageMaker Pipelines (Processing/HPO/Train/Eval)
                                          |
                                          v
                                    SageMaker Model Registry
                                          |
                                          v (Terraform per environment)
                                    SageMaker Endpoint (Staging) --> SageMaker Endpoint (Prod)
                                                                          |
                                                                          v
                                                                    SageMaker Model Monitor
                                                                          |
                                                                          v
                                                                    CloudWatch --> EventBridge --> Retrain
```

**Components to manage**: Snowflake, S3, SageMaker (Notebooks, Pipelines, Registry, Endpoints, Model Monitor), Terraform, CloudWatch, EventBridge, IAM roles per environment.

### Proposed State (Snowflake)

```
Snowflake DEV                          Snowflake STAGING                   Snowflake PROD
+---------------------------+          +------------------+                +---------------------------+
| Feature Store (DT)        |          | Model Registry   |                | Model Registry            |
| Notebooks (Container RT)  |   RBAC   | Staging Endpoint |    RBAC        | Production Endpoint (SPCS)|
| Model Registry            | -------> | (Validation)     | ------------>  | ML Observability          |
| Experiment Tracking       |          +------------------+                | Alerts + Tasks (Retrain)  |
+---------------------------+                                              +---------------------------+
                                                                                     |
                                                                               Taktile (REST API)
```

**Components to manage**: Snowflake (single platform).

### Component Mapping

| # | SageMaker Component | Snowflake Equivalent | Operational Delta |
|---|---|---|---|
| 1 | SageMaker Studio Notebooks | Snowflake Notebooks (Container Runtime) | No separate IDE; native SQL+Python in one UI |
| 2 | SageMaker Processing Step | Feature Store (Dynamic Tables) | Incremental refresh, no S3 staging |
| 3 | SageMaker HyperParameter Tuning | `snowflake.ml.modeling.tune.Tuner` | Bayesian optimization, distributed across compute nodes |
| 4 | SageMaker Train Step | Notebook / Stored Procedure | Same XGBoost code, runs on warehouse or GPU pool |
| 5 | SageMaker Evaluation Step | In-notebook evaluation + metric logging | Metrics stored as model version metadata |
| 6 | SageMaker Condition Step | Python logic in notebook / Task DAG | Threshold checks before registration |
| 7 | SageMaker Model Registry + approval | Snowflake Model Registry per env + RBAC | Schema-level objects, promotion is copy + RBAC gate |
| 8 | SageMaker Endpoint (Terraform per env) | SPCS Model Serving (`create_service`) | No Terraform; 1-line Python API, per-env services |
| 9 | SageMaker Model Monitor + CloudWatch + EventBridge | ML Observability + Snowflake Alerts | Single platform, no cross-service wiring |
| 10 | AWS accounts / Terraform workspaces | Database-level isolation + RBAC roles | No multi-account infra; SQL-managed promotion |

---

## Environment Strategy

### DEV / STAGING / PROD Separation

Environments are separated at the database level with RBAC controlling promotion:

| Environment | Database | Schemas | Purpose | Access |
|---|---|---|---|---|
| DEV | `PD_DEMO_DEV` | `ML`, `FEATURE_STORE`, `REGISTRY` | Experimentation, training, HPO | DS: read/write |
| STAGING | `PD_DEMO_STAGING` | `REGISTRY`, `SERVING` | Model validation, integration testing | DS: read-only, ML Ops: read/write |
| PROD | `PD_DEMO_PROD` | `REGISTRY`, `SERVING`, `MONITORING` | Live inference, monitoring | ML Ops: read/write, DS: read-only |

### RBAC Role Hierarchy

```
ACCOUNTADMIN
    |
PD_MLOPS (ML Operations)
    |-- Full access: STAGING, PROD
    |-- Read access: DEV
    |
PD_DS_DEV (Data Scientists)
    |-- Full access: DEV
    |-- Read access: STAGING, PROD
```

### Promotion Workflow

```
1. DS trains model in DEV --> logs to DEV Registry
2. DS requests promotion (or automated via Task)
3. ML Ops validates in STAGING:
   - Copy model artifact from DEV to STAGING Registry
   - Deploy staging endpoint
   - Run validation suite on held-out data
   - Assert metrics match DEV (AUC, Gini within tolerance)
4. ML Ops promotes to PROD:
   - Copy model to PROD Registry
   - Deploy production endpoint
   - Configure Model Monitor
5. Ongoing: Model Monitor detects drift --> Alert fires --> Task retrains in DEV
```

### Comparison with SageMaker Environment Separation

| Aspect | SageMaker | Snowflake |
|---|---|---|
| Isolation mechanism | Separate AWS accounts + IAM | Database-level + RBAC roles |
| Infrastructure per env | Terraform workspace per env | SQL `CREATE DATABASE` |
| Promotion mechanism | Manual approval in SM Registry + Terraform deploy | `log_model()` across registries, RBAC-gated |
| Cost of additional env | Full SageMaker endpoint cost per env | Zero-copy clone for data; endpoint only in STAGING/PROD |
| Time to create new env | Hours (Terraform + IAM + networking) | Minutes (SQL + GRANT statements) |

---

## Model Specification

| Parameter | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Target | `DEFAULT_90DPM` (binary: 1 = default within 90 days past maturity) |
| Default rate | ~5-8% (realistic for BNPL) |
| Training data | 500,000 credit applications (~400k unique customers, 18 months of data) |
| Inference volume | 600,000 records (300k normal + 300k drifted, ~10k/day) |
| Model features | **20 total** (17 numeric bureau/application + 3 one-hot channel) |
| Bureau/app features (17) | num_credit_products, num_inactive_accounts, num_credit_searches_l6m, total_outstanding_balance, max_delinquency_days, credit_utilisation_ratio, months_since_oldest_account, num_defaults_l3y, num_ccjs, credit_score, num_open_accounts, total_credit_limit, num_missed_payments_l12m, debt_to_income_ratio, months_since_last_default, num_hard_searches_l3m, applicant_age_years |
| Channel features (3) | Origination channel: Direct, Google, Meta (one-hot encoded) |
| Preprocessing | Missing value imputation (median), categorical encoding (one-hot in Feature Store), outlier capping |
| Evaluation metrics | AUC, Gini coefficient, KS statistic, calibration (Brier score), precision-recall |
| Inference latency target | ~50ms (point-of-application decisioning) |
| Consumer | Taktile decisioning platform via REST API |

---

## Demo Walkthrough

### Time Budget (55 minutes)

| Section | Notebook | Duration | Key Message |
|---|---|---|---|
| Introduction | -- | 5 min | Problem statement, current vs proposed architecture, env strategy |
| Data and Features | `nb01_data_and_features.ipynb` | 12 min | 3-env setup, Feature Store eliminates data movement |
| Training, Registry, Promotion | `nb02_training_and_registry.ipynb` | 12 min | Same code, DEV->STAGING->PROD promotion, RBAC governance |
| Serving and Latency | `nb03_serving_and_inference.ipynb` | 12 min | ~50ms inference from PROD, no Terraform, Taktile integration |
| Monitoring and Drift | `nb04_monitoring_and_drift.ipynb` | 9 min | PROD monitoring replaces 3 AWS services with 1 Snowflake feature |
| Wrap-up and Q&A | -- | 5 min | Cost comparison, migration effort, next steps |

### Notebook 1: Environment Setup, Data, and Feature Engineering

**Purpose**: Set up multi-environment infrastructure, generate realistic synthetic BNPL data (500k applications, 20 features), and build a Feature Store.

**Key cells**:
1. SQL: Create 3 databases (DEV/STAGING/PROD), schemas, warehouses, RBAC roles
2. SQL: Generate synthetic `CREDIT_APPLICATIONS` table (500k rows, 20 features) with credit bureau features
3. Python: Data quality exploration -- distributions, correlations, class imbalance
4. Python: Create Feature Store with entity and managed feature views (Dynamic Tables)
5. Python: Preprocessing pipeline -- imputation, encoding, scaling
6. Python: Generate training dataset with point-in-time correctness (ASOF JOIN)

**Talk track**: "Your data already lives in Snowflake. We set up three environments with one SQL script -- no Terraform, no AWS Organizations. The Feature Store eliminates S3 staging entirely. Dynamic Tables give you incremental refresh. The ASOF JOIN for training data prevents data leakage -- critical for regulatory compliance."

### Notebook 2: Model Training, Registry, and Environment Promotion

**Purpose**: Train XGBoost with HPO, evaluate comprehensively, register in DEV, promote through environments.

**Key cells**:
1. Python: Train/validation/test split (70/15/15)
2. Python: Baseline XGBoost + evaluation
3. Python: Hyperparameter tuning (Bayesian optimization, 30 trials)
4. Python: Full evaluation: ROC, PR curve, SHAP, calibration plot, Gini, KS
5. Python: Conditional registration (AUC > threshold)
6. Python: Log model to DEV Registry with metrics
7. Python: Promote DEV -> STAGING (load + re-register, RBAC-gated)
8. Python: Validate on staging (batch inference, assert metrics match)
9. Python: Promote STAGING -> PROD
10. SQL: Show models across all three environments

**Talk track**: "Same XGBoost code you write today. The Tuner gives you Bayesian HPO natively. The Model Registry is a first-class Snowflake object -- RBAC controls who can register vs who can deploy. Watch the promotion flow: DEV -> STAGING with validation -> PROD. In SageMaker this requires Terraform to spin up separate endpoints per environment."

### Notebook 3: Model Serving and Real-Time Inference

**Purpose**: Deploy from PROD registry as REST endpoint, benchmark latency, show Taktile integration.

**Key cells**:
1. Python: Deploy model from PROD registry to SPCS (`create_service`)
2. SQL: Verify service running
3. Python: Single-record inference -- measure latency (target ~50ms)
4. Python: Batch inference benchmark (1000 records)
5. Python: Taktile integration pattern -- REST API structure, auth, error handling
6. SQL: Batch inference via SQL (for offline scoring / strategy backtesting)
7. Python: Cost/latency comparison vs SageMaker

**Talk track**: "One line of Python to deploy a production endpoint. No Terraform, no ECR, no IAM roles. The REST API is compatible with any HTTP client -- Taktile can call it exactly like they call SageMaker today. And the cost model is fundamentally different: SPCS charges per second of compute, while SageMaker charges for always-on instances."

### Notebook 4: Model Monitoring and Drift Detection

**Purpose**: Set up monitoring in PROD, simulate drift, demonstrate alerting and retraining triggers.

**Key cells**:
1. SQL: Create inference log table in PROD
2. SQL: Insert simulated inference data (normal + drifted periods)
3. SQL: Create Model Monitor with segmentation by origination channel
4. Python: Query and visualize drift metrics (PSI per feature over time)
5. Python: Query performance metrics (AUC over time by channel)
6. SQL: Create Alert for drift-triggered retraining
7. SQL: Create Task DAG for automated retrain pipeline
8. Python: Comparison summary -- SageMaker Monitor vs Snowflake ML Observability

**Talk track**: "This replaces your entire CloudWatch + EventBridge + SageMaker Monitor stack. The Model Monitor computes PSI and performance metrics automatically. Segmentation by origination channel means you catch drift in Google vs Meta vs Direct cohorts independently. The Alert -> Task chain replaces EventBridge -> SageMaker Pipelines."

---

## Answering Their Key Questions

From the customer's "What We Want to See" slide:

### 1. How the above pipelines could be implemented with Snowflake with minimal disruption

The component mapping table above shows a 1:1 replacement for every SageMaker component. The XGBoost training code is identical -- the same `xgboost` library runs in Snowflake Notebooks with Container Runtime. The key change is replacing SageMaker-specific orchestration (Pipelines, Processing Steps) with Snowflake-native equivalents (Feature Store, Tasks). Data preparation code that currently writes to S3 is eliminated entirely since data stays in Snowflake.

### 2. Focus on connectivity with Platform and other apps

Taktile integration is straightforward:
- **REST API**: The SPCS model serving endpoint exposes a standard HTTP API. Taktile calls it exactly like it calls SageMaker endpoints today.
- **Authentication**: Programmatic Access Tokens (PAT) or OAuth.
- **Request/Response**: Standard JSON format, compatible with any HTTP client.
- **Network**: Ingress can be enabled on the service for external access, with network policies controlling IP allowlisting.

For batch/offline scoring (e.g., strategy backtesting), the model can be called directly via SQL -- no API call needed.

### 3. Why we think the Snowflake architecture would be better

**Operational simplicity**:
- 1 platform vs 8+ AWS services
- No Terraform for ML infrastructure
- Environment promotion via RBAC + SQL, not IaC
- Unified monitoring (no CloudWatch/EventBridge wiring)

**Cost**:
- Eliminate always-on SageMaker endpoint costs (~$167/month per env for ml.m5.xlarge)
- SPCS per-second billing with auto-suspend -- ideal for bursty BNPL application traffic
- No S3 data staging costs
- No data transfer costs between Snowflake and SageMaker

**Performance**:
- XGBoost on SPCS CPU achieves ~50ms inference latency for credit risk payloads
- Zero data movement means faster feature freshness and training iteration
- Feature Store with Dynamic Tables provides incremental refresh

**Governance**:
- RBAC on model objects -- DS can develop, only ML Ops can promote to production
- Model lineage tracks which features and training data produced each version
- Point-in-time correctness built into Feature Store (regulatory compliance)

### 4. Evaluate benefits vs effort/cost of migration

| Migration Step | Effort | Notes |
|---|---|---|
| Set up Snowflake ML infrastructure | 1 day | SQL script creates databases, schemas, roles, compute pool |
| Port training code to Snowflake Notebook | 1-2 days | XGBoost code is identical; replace SageMaker-specific orchestration |
| Set up Feature Store | 2-3 days | Define entities, feature views, validate against existing feature pipeline |
| Deploy model serving endpoint | 1 day | `create_service()` + REST API testing with Taktile |
| Set up monitoring and alerting | 1 day | Model Monitor + Alerts + Task DAG |
| Validate end-to-end | 2-3 days | Integration testing, latency benchmarks, failover testing |
| **Total** | **~2 weeks** | Assumes parallel work by 2 DS/ML Ops engineers |

### 5. Cortex spotlight

The demo shows Cortex AI in two ways:
- Built-in `SNOWFLAKE.ML.CLASSIFICATION` can be run as a no-code baseline in 2 minutes, demonstrating platform breadth
- Snowflake Notebooks with Container Runtime support any Python ML library (XGBoost, scikit-learn, PyTorch, etc.)

### 6. Model monitoring & drift detection spotlight

Notebook 4 provides a comprehensive demonstration:
- PSI (Population Stability Index) per feature with time-series visualization
- Performance metrics (AUC) segmented by origination channel
- Automated drift alerting via Snowflake Alerts
- Drift-triggered retraining via Task DAGs
- Direct comparison with SageMaker Model Monitor + CloudWatch + EventBridge

---

## Additional Capabilities (Not in Current Architecture)

These are capabilities that Snowflake provides beyond what the current SageMaker architecture offers:

1. **Point-in-time Feature Correctness**: The Feature Store's ASOF JOIN prevents data leakage in training data generation. This is typically handled manually (and error-prone) in SageMaker pipelines.

2. **SHAP Explainability**: Built-in SHAP analysis on registered models. Critical for BNPL regulatory compliance -- adverse action notices require feature-level explanations.

3. **Zero-Copy Cloning**: `CREATE DATABASE PD_DEMO_STAGING CLONE PD_DEMO_DEV` gives a full copy of DEV data in STAGING at zero additional storage cost. No equivalent in SageMaker.

4. **Model Lineage**: Track exactly which features and training data produced each model version. Available via `mv.lineage()` API.

5. **SQL-Native Batch Inference**: Run predictions directly in SQL pipelines, Dynamic Tables, or dbt models without deploying an endpoint.

6. **CI/CD with Snowflake CLI**: The `snow` CLI automates notebook deployment, model promotion, and service creation -- can replace Terraform for the ML workflow.

---

## Deployment Instructions

### Prerequisites

- Snowflake account with `ACCOUNTADMIN` access (for initial setup)
- `snow` CLI installed (`pip install snowflake-cli-labs`)
- Cortex Code CLI (for notebook building)

### Step 1: Run Setup Script

```bash
cd /path/to/prob_default_ml
snow sql -f scripts/setup.sql
```

### Step 2: Deploy Notebooks

```bash
snow stage copy notebooks/nb01_data_and_features.ipynb @PD_DEMO_DEV.ML.NOTEBOOKS/
snow stage copy notebooks/nb02_training_and_registry.ipynb @PD_DEMO_DEV.ML.NOTEBOOKS/
snow stage copy notebooks/nb03_serving_and_inference.ipynb @PD_DEMO_DEV.ML.NOTEBOOKS/
snow stage copy notebooks/nb04_monitoring_and_drift.ipynb @PD_DEMO_DEV.ML.NOTEBOOKS/

snow notebook create NB01_DATA_AND_FEATURES --from-stage @PD_DEMO_DEV.ML.NOTEBOOKS/nb01_data_and_features.ipynb --database PD_DEMO_DEV --schema ML
snow notebook create NB02_TRAINING_AND_REGISTRY --from-stage @PD_DEMO_DEV.ML.NOTEBOOKS/nb02_training_and_registry.ipynb --database PD_DEMO_DEV --schema ML
snow notebook create NB03_SERVING_AND_INFERENCE --from-stage @PD_DEMO_DEV.ML.NOTEBOOKS/nb03_serving_and_inference.ipynb --database PD_DEMO_DEV --schema ML
snow notebook create NB04_MONITORING_AND_DRIFT --from-stage @PD_DEMO_DEV.ML.NOTEBOOKS/nb04_monitoring_and_drift.ipynb --database PD_DEMO_DEV --schema ML
```

### Step 3: Open in Snowsight

Navigate to Snowsight > Notebooks and run each notebook sequentially.

---

## File Structure

```
prob_default_ml/
  README.md                                    <-- This design document
  notebooks/
    nb01_data_and_features.ipynb               <-- Notebook 1: Env setup + Data + Feature Store
    nb02_training_and_registry.ipynb           <-- Notebook 2: Training + Registry + Promotion
    nb03_serving_and_inference.ipynb           <-- Notebook 3: Serving + Latency + Taktile
    nb04_monitoring_and_drift.ipynb            <-- Notebook 4: Monitoring + Drift + Alerting
  scripts/
    setup.sql                                  <-- SQL setup for all 3 environments
  docs/
    talk_track.md                              <-- Detailed presenter notes
    Snowflake Demo (Zilch side).pdf            <-- Original customer architecture deck
```
