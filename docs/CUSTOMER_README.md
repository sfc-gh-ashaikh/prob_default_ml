# Probability of Default: End-to-End ML on Snowflake

## Demo Overview

This demo walks through a complete Probability of Default (PD) model lifecycle on Snowflake -- from feature engineering through to production serving and automated monitoring -- as an alternative to a SageMaker-based architecture.

The scenario is modelled on a real-world BNPL (Buy Now, Pay Later) credit risk pipeline: an XGBoost classifier trained on credit bureau features, deployed as a real-time endpoint for point-of-application decisioning, and monitored for feature drift.

### Objectives

1. **Show a 1:1 mapping** of every SageMaker pipeline component to a Snowflake equivalent, using the same XGBoost code and credit risk feature set
2. **Benchmark real-time inference** against the ~50ms latency target required for point-of-application decisioning
3. **Demonstrate native preprocessing** -- categorical encoding and missing value imputation handled directly in Snowflake, not in external pipelines
4. **Walk through DEV/STAGING/PROD environment promotion** with RBAC governance, replacing Terraform-managed infrastructure
5. **Show drift detection and automated retraining** as a single-platform capability, replacing the SageMaker Monitor + CloudWatch + EventBridge stack
6. **Quantify cost and operational simplicity gains** at each stage of the pipeline

### What is in the demo

| Notebook | What it covers | Duration |
|---|---|---|
| **01 - Data & Features** | Environment setup (DEV/STAGING/PROD), 500k synthetic credit applications, Feature Store with 20 model features, point-in-time correct training data | 12 min |
| **02 - Training & Registry** | XGBoost training, Bayesian HPO (30 trials), SHAP explainability, conditional registration, promotion DEV -> STAGING -> PROD | 12 min |
| **03 - Serving & Inference** | SPCS real-time endpoint deployment, latency benchmark, decisioning platform integration pattern, SQL-native batch inference | 12 min |
| **04 - Monitoring & Drift** | 600k inference records, Model Monitor with PSI drift detection, performance tracking by origination channel, automated retraining alerts | 9 min |

---

## Architecture Comparison

### Current: SageMaker-Based Pipeline

```
                            AWS VPC
 ┌─────────────────────────────────────────────────────────────────────┐
 │                                                                     │
 │  Snowflake ──► S3 ──► SageMaker Pipelines                          │
 │  (Data)       (Staging)  ├─ Processing Step (feature engineering)   │
 │                          ├─ HPO Step (hyperparameter tuning)        │
 │                          ├─ Train Step                              │
 │                          ├─ Evaluation Step                         │
 │                          └─ Condition Step (threshold gate)         │
 │                                    │                                │
 │                                    ▼                                │
 │                          SageMaker Model Registry                   │
 │                                    │                                │
 │                          ┌─────────┴─────────┐                     │
 │                          ▼                   ▼                      │
 │                    Terraform            Terraform                   │
 │                    (staging)            (production)                 │
 │                          ▼                   ▼                      │
 │                    SM Endpoint          SM Endpoint ◄── Taktile     │
 │                                              │                      │
 │                                    SM Model Monitor                 │
 │                                              │                      │
 │                                    CloudWatch ──► EventBridge       │
 │                                                       │             │
 │                                              Retrain trigger        │
 └─────────────────────────────────────────────────────────────────────┘

 Services to manage: Snowflake, S3, SageMaker (5 sub-services),
                     Terraform, CloudWatch, EventBridge, IAM
```

### Proposed: Consolidated Snowflake Pipeline

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Snowflake                                                          │
 │                                                                     │
 │  DEV                      STAGING                 PROD              │
 │  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐  │
 │  │ Model        │        │ Model        │  RBAC  │ Model        │  │
 │  │ Registry     │──────►│ Registry     │──────►│ Registry     │  │
 │  │              │        │              │        │              │  │
 │  │ Notebooks    │        │ Validation   │        │ SPCS Endpoint│◄─── Taktile
 │  │ (Container)  │        │ Endpoint     │        │              │  │
 │  │              │        └──────────────┘        │ Model Monitor│  │
 │  │ Feature Store│                                │              │  │
 │  │ (Dynamic Tbl)│        ◄───────────────────────│ Alert ► Task │  │
 │  └──────────────┘           drift-triggered       │ (retrain)   │  │
 │                              retraining           │              │  │
 │                                                   │ Sched. Task  │  │
 │  Shadow Testing: MODEL(V1) vs MODEL(V2) in SQL    │ (weekly)    │  │
 │  (no shadow endpoint infra needed)                └──────────────┘  │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘

 Services to manage: Snowflake (single platform)
```

### Component-by-Component Mapping

| SageMaker Component | Snowflake Equivalent | What Changes |
|---|---|---|
| SageMaker Studio Notebooks | Snowflake Notebooks (Container Runtime) | Native SQL + Python in one UI. No separate notebook instance to manage. |
| SageMaker Processing Step | Feature Store (Dynamic Tables) | Incremental refresh -- only changed data is reprocessed. No S3 staging. |
| SageMaker HyperParameter Tuning | Optuna in-notebook / Snowflake Tuner API | Same Bayesian optimisation. No separate Tuning Job to orchestrate. |
| SageMaker Train Step | Same XGBoost code in Notebook | Identical training code. Runs on warehouse compute with auto-suspend. |
| SageMaker Evaluation Step | In-notebook evaluation + metric logging | Metrics stored as model version metadata. No separate evaluation pipeline. |
| SageMaker Condition Step | Python threshold check before `log_model()` | Same logic, no pipeline orchestration overhead. |
| SageMaker Model Registry | Snowflake Model Registry (per environment) | Schema-level objects with RBAC. Promotion is a Python API call, not Terraform. |
| SageMaker Endpoint (via Terraform) | SPCS Model Serving (`create_service()`) | 1 Python call. No Terraform, no ECR, no IAM roles. Per-second billing. |
| SageMaker Model Monitor + CloudWatch + EventBridge | ML Observability + Snowflake Alerts + Tasks | 1 SQL statement replaces 3 AWS services. No cross-service wiring. |
| SageMaker Shadow Testing | Multi-version SQL scoring (`MODEL(V1)` vs `MODEL(V2)` in same query) | No shadow endpoint infra. Compare N versions simultaneously in a single query. |
| EventBridge scheduled retraining rule | Snowflake Task with CRON schedule | Same warehouse, same RBAC. Coexists with drift-triggered alerts. |
| AWS accounts + Terraform workspaces (env separation) | Database isolation + RBAC roles | Environment setup in minutes via SQL, not hours via Terraform. |

---

## Performance Considerations

### Inference Latency

| Factor | Detail |
|---|---|
| **Target** | ~50ms per inference at point of application |
| **Model profile** | XGBoost classifier, 20 input features, binary output |
| **Serving infrastructure** | SPCS (Snowpark Container Services) on CPU_X64_S compute pool |
| **Expected performance** | XGBoost on CPU typically achieves 10-50ms for small models with compact payloads. Benchmarked live in Notebook 3. |
| **Scaling** | Horizontal autoscaling via min/max instances. Service scales automatically under load. |
| **Comparison** | SageMaker ml.m5.large achieves ~50-100ms for similar XGBoost payloads. Comparable latency profile. |

### Batch Inference

Snowflake provides SQL-native batch inference that has no SageMaker equivalent without deploying a separate Batch Transform job:

```sql
SELECT APPLICATION_ID,
       MODEL(PD_XGBOOST, V1)!PREDICT_PROBA(...):output_feature_1::FLOAT AS pd_score
FROM CREDIT_APPLICATIONS;
```

This enables strategy backtesting, portfolio scoring, and continuous scoring pipelines (via Dynamic Tables) without a real-time endpoint.

---

## Cost Levers

### Infrastructure Eliminated

*Based on actual reported SageMaker costs: ml.m5.large @ $0.128/hr endpoint, ~2.5 hr @ $1/hr training. Snowflake pricing at Enterprise edition (~$3/credit).*

| Cost Component | SageMaker (actual) | Snowflake | Saving |
|---|---|---|---|
| **Real-time endpoint** (staging + prod) | ~$184/month (2x ml.m5.large @ $0.128/hr, 24/7) | Per-second billing; staging effectively free; prod depends on traffic pattern | ~$87-140/month (staging elimination + bursty savings) |
| **S3 data staging** | Storage + PUT/GET + cross-service transfer | Eliminated (data stays in Snowflake) | ~$10-50/month depending on volume |
| **Terraform/IaC management** | Engineering time for HCL, state management, plan/apply per env | SQL scripts + `snow` CLI | Hours per deployment saved |
| **CloudWatch metrics** | Metric ingestion + dashboard + alarm evaluation | Included in warehouse compute | ~$10-30/month |
| **EventBridge rules** | Rule evaluation costs | Snowflake Alert (uses existing warehouse) | Marginal but removes a service |
| **SageMaker notebook instance** | ~$34/month (ml.t3.medium, 24/7) | Warehouse auto-suspend (60s idle timeout) | ~$34/month |
| **IAM / cross-account setup** | Engineering time per environment | RBAC via SQL GRANT statements | Hours per environment saved |

### Compute Cost Model

| Aspect | SageMaker (actual) | Snowflake | Notes |
|---|---|---|---|
| **Training compute** | ~$2.50/run (2.5 hr @ $1/hr) | ~$4-5/run (MEDIUM WH @ $12/hr, ~20 min active) | SageMaker cheaper per-run. Snowflake auto-suspends between HPO trials. |
| **Serving compute** | $92/month per endpoint (ml.m5.large @ $0.128/hr, 24/7) | $130/month (CPU_X64_XS @ $0.18/hr, 24/7) or ~$43/month (8hr/day bursty) | Bursty traffic favours Snowflake. 24/7 slightly favours SageMaker. |
| **Feature engineering** | Full SageMaker Processing job every run | Dynamic Table incremental refresh (only changed rows) | Snowflake only reprocesses new/changed data. |
| **Monitoring compute** | Separate SageMaker Monitor scheduling job | Uses existing warehouse, auto-suspends between refreshes | No additional service cost. |
| **HPO compute** | Included in training run cost | Runs on same notebook compute (or distributed via Tuner API) | Both included in training estimate. |

### Operational Simplicity

| Metric | SageMaker Architecture | Snowflake Architecture |
|---|---|---|
| AWS services to manage | 8+ (SageMaker, S3, Terraform, CloudWatch, EventBridge, IAM, ECR, VPC) | 1 (Snowflake) |
| Environment setup time | Hours (Terraform + IAM + networking) | Minutes (SQL + GRANT) |
| Model promotion workflow | Terraform apply + ECR build + endpoint deploy per env | `log_model()` across registries (seconds) |
| Monitoring configuration | Model Monitor config + CloudWatch namespace + alarm + EventBridge rule | 1 SQL statement (`CREATE MODEL MONITOR`) |
| Drift-triggered retraining | CloudWatch Alarm -> EventBridge Rule -> SageMaker Pipeline | Alert -> Task (2 SQL objects) |
| Identity & access | IAM roles per account + STS assume-role chains | RBAC roles with SQL GRANT |

### Total Cost of Ownership: Honest Assessment

*This analysis uses actual reported SageMaker costs and verified Snowflake pricing. We are deliberately transparent about where SageMaker is cheaper and where Snowflake wins.*

#### Snowflake Pricing Basis

| Resource | Credit rate | Cost @ Enterprise (~$3/credit) |
|---|---|---|
| SPCS CPU_X64_XS (1 vCPU, 6 GiB) | 0.06 credits/hr/node | $0.18/hr |
| SPCS CPU_X64_S (3 vCPU, 8 GiB) | 0.11 credits/hr/node | $0.33/hr |
| MEDIUM virtual warehouse | 4 credits/hr | $12/hr |

*Credit rates sourced from the [Snowflake Service Consumption Table](https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf). Actual credit cost depends on your contract -- capacity commitments reduce the per-credit rate by 15-40%.*

#### Monthly Infrastructure Cost

| Component | SageMaker (actual) | Snowflake (estimate) | Delta |
|---|---|---|---|
| Prod endpoint (24/7) | $92/month (ml.m5.large @ $0.128/hr) | $130/month (CPU_X64_XS @ $0.18/hr, 24/7) | +$38/month |
| Prod endpoint (8 hr/day -- bursty BNPL) | $92/month (always-on, no per-hour option) | $43/month (CPU_X64_XS @ $0.18/hr x 8 hrs x 30 days) | -$49/month |
| Staging endpoint | $92/month (always-on) | ~$5/month (only active during validation) | -$87/month |
| Training (4 retrains/month) | $10/month (4 x $2.50) | $16-20/month (4 x $4-5 on MEDIUM WH) | +$6-10/month |
| S3 staging + data transfer | $10-50/month | $0 (eliminated) | -$10 to -$50/month |
| CloudWatch + EventBridge | $10-30/month | $0 (included in warehouse compute) | -$10 to -$30/month |
| Notebook instance (ml.t3.medium) | $34/month (24/7) | $0 (warehouse auto-suspend) | -$34/month |
| **Total (24/7 endpoint)** | **~$250-310/month** | **~$150-175/month** | **-$75 to -$160/month** |
| **Total (bursty 8hr/day endpoint)** | **~$250-310/month** | **~$65-90/month** | **-$160 to -$245/month** |

#### Where SageMaker is Cheaper

- **Training compute per run**: $2.50 vs ~$4-5 on Snowflake. SageMaker's $1/hr ML instances are hard to beat on raw unit cost. This is an honest ~2x premium on per-run training compute. However, Snowflake's auto-suspend means you only pay for active compute time (no manual instance shutdown between HPO trials).
- **Always-on prod endpoint**: If the endpoint must be available 24/7 with zero cold-start tolerance, SageMaker ml.m5.large at $0.128/hr ($92/month) is cheaper than SPCS CPU_X64_XS at $0.18/hr ($130/month). This gap narrows with a Snowflake capacity commitment (lower per-credit rate) and is reversed if traffic is bursty.

#### Where Snowflake is Cheaper

- **Bursty endpoint pattern**: BNPL application traffic peaks during business hours and drops overnight/weekends. SPCS per-second billing with service suspension during off-hours = ~$43/month vs SageMaker's always-on $92/month (53% saving).
- **Staging endpoint**: In SageMaker, a staging endpoint costs the same as production ($92/month always-on). In Snowflake, the staging SPCS service is only active during model validation -- effectively free. This alone saves ~$87/month.
- **S3 data staging**: Eliminated entirely. Data never leaves Snowflake. Saves $10-50/month in storage, PUT/GET, and data transfer costs.
- **Monitoring stack**: CloudWatch metrics, alarms, and EventBridge rules are replaced by Snowflake objects that run on the existing warehouse (auto-suspend between refreshes). Saves $10-30/month.
- **Notebook instance**: SageMaker charges for always-on notebook instances ($34/month). Snowflake warehouse compute auto-suspends after 60 seconds of inactivity.

#### Where the Biggest Saving Is (not in the numbers above)

Engineering time is the largest hidden cost that does not appear in the infrastructure comparison:

| Activity | SageMaker | Snowflake |
|---|---|---|
| Endpoint deployment | Terraform HCL + ECR image build + IAM role per env | `create_service()` -- 1 Python call |
| Environment promotion | Terraform plan/apply per environment | `log_model()` -- seconds |
| Monitoring setup | Model Monitor config + CloudWatch namespace + alarm + EventBridge rule | `CREATE MODEL MONITOR` -- 1 SQL statement |
| New model version rollout | Terraform apply + ECR rebuild + endpoint update | `create_service()` with new version |
| Access control changes | IAM policy updates across accounts | `GRANT` / `REVOKE` SQL |

At a conservative estimate of 2-4 hours of engineering time per deployment cycle at ~$100/hr blended rate, this represents **$200-400 per deployment** in hidden costs that Snowflake eliminates.

#### Annual Summary

| | SageMaker | Snowflake (bursty) | Annual Saving |
|---|---|---|---|
| Infrastructure | ~$3,000-3,700/year | ~$780-1,080/year | ~$1,900-2,900/year |
| Engineering time (est. 12 deployments/year) | ~$2,400-4,800/year | Minimal | ~$2,400-4,800/year |
| **Total estimated** | **~$5,400-8,500/year** | **~$780-1,080/year** | **~$4,300-7,400/year** |

*Notes:*
- *Snowflake costs assume Enterprise edition @ ~$3/credit. Capacity commitments reduce this to ~$2-2.50/credit, lowering all Snowflake estimates by 17-33%.*
- *These estimates are for a single PD model pipeline. For organisations running multiple models, the savings scale linearly on infrastructure and super-linearly on engineering time (shared platform, shared RBAC, shared monitoring).*
- *SPCS endpoint cost depends on traffic pattern. The bursty estimate assumes service suspension during 16 off-peak hours/day via scheduled Tasks.*

---

## Model Specification

| Parameter | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Target | 90 DPM (Days Past Maturity) -- binary classification |
| Features | 20 total: 17 credit bureau / application features + 3 origination channel (one-hot) |
| Training data | 500,000 credit applications (~400k unique customers) |
| Inference data | 600,000 records (~10k/day over 60 days) |
| Preprocessing | Missing value imputation (median), categorical encoding (one-hot in Feature Store), outlier capping |
| Evaluation | AUC, Gini, KS statistic, calibration (Brier score), SHAP explainability |
| Serving target | ~50ms real-time inference for point-of-application decisioning |

### Feature Set (20 features)

**Credit bureau features (10):** num_credit_products, num_inactive_accounts, num_credit_searches_l6m, total_outstanding_balance, max_delinquency_days, credit_utilisation_ratio, months_since_oldest_account, num_defaults_l3y, num_ccjs, credit_score

**Application features (7):** num_open_accounts, total_credit_limit, num_missed_payments_l12m, debt_to_income_ratio, months_since_last_default, num_hard_searches_l3m, applicant_age_years

**Origination channel (3):** channel_direct, channel_google, channel_meta

---

## Additional Capabilities Beyond Current Architecture

These are capabilities Snowflake provides that extend beyond the current SageMaker pipeline:

### Demonstrated in This Demo

| Capability | What it does | Why it matters | SageMaker equivalent |
|---|---|---|---|
| **Point-in-time Feature Correctness** | Feature Store ASOF JOIN ensures no future data leaks into training | Regulatory compliance for credit risk | Requires manual timestamp handling in processing step |
| **SHAP Explainability** | Built-in `explain()` method on registered models + in-notebook SHAP | Adverse action notices require feature-level explanations in regulated BNPL lending | Requires SageMaker Clarify (separate job + S3 config) |
| **Zero-Copy Cloning** | `CLONE` a database for STAGING testing at zero storage cost | Instant environment replication for integration testing | No equivalent -- must copy data to new S3 prefix |
| **SQL-Native Batch Inference** | Call model predictions directly in SQL queries | Strategy backtesting as a SQL query, no Batch Transform job | Requires separate SageMaker Batch Transform job |
| **Shadow Testing via SQL** | Score with multiple model versions in a single query, compare instantly | Validate candidate models on real data before promotion | Requires SageMaker Shadow Testing (Terraform + S3 capture + custom analysis) |
| **Model Lineage** | Track which features and training data produced each model version | Audit trail for FCA/PRA regulatory reporting | Manual tracking or SageMaker ML Lineage Tracking (limited) |
| **Segmented Monitoring** | Monitor drift per origination channel (or any dimension) with `SEGMENT_COLUMNS` | Catch degradation in specific cohorts that overall metrics miss | Custom analysis code required per segment |
| **Dual Retraining Triggers** | Drift-triggered (Alert) + scheduled (Task with CRON) coexist natively | Both reactive and proactive model freshness | Requires separate EventBridge rules + CloudWatch alarms for each trigger |

### Additional Snowflake Capabilities (not in SageMaker)

| Capability | What it does | Why it matters for BNPL PD | SageMaker alternative |
|---|---|---|---|
| **Continuous Scoring via Dynamic Tables** | `CREATE DYNAMIC TABLE AS SELECT MODEL(...)!PREDICT_PROBA(...)` -- auto-refreshes as new data arrives | New applications scored automatically without a batch job or endpoint call | No equivalent -- requires Batch Transform or always-on endpoint |
| **Cortex LLM for Model Interpretation** | Use Cortex `COMPLETE()` to generate natural-language summaries of SHAP outputs | Automated adverse action notice text generation from model explanations | No equivalent -- requires external LLM integration (Bedrock + Lambda) |
| **Snowflake Alerts with Email/Webhook** | Drift alerts can notify via email or webhook in addition to triggering tasks | Ops team gets notified before automated retraining runs | Requires CloudWatch Alarm + SNS topic + subscription configuration |
| **Git Integration for ML Code** | Notebooks and SQL scripts version-controlled via Snowflake Git integration | Code versioning, review, and rollback for ML pipelines | Requires CodeCommit/GitHub + SageMaker Projects + CI/CD pipeline |
| **Time Travel for Data Debugging** | Query historical table state with `AT(TIMESTAMP => ...)` or `BEFORE(STATEMENT => ...)` | Investigate data quality issues, reproduce past training runs exactly | No equivalent -- requires versioned S3 prefixes and manual management |
| **Data Sharing for Model Output Distribution** | Share model predictions with downstream teams via Snowflake Secure Data Sharing | Credit risk scores available to strategy, collections, and finance teams instantly | Requires S3 cross-account access + IAM configuration |
| **Snowpark Container Services (SPCS) GPU** | Deploy GPU-accelerated models (future deep learning) on the same platform | Upgrade path from XGBoost to neural networks without changing infrastructure | Requires new SageMaker endpoint configuration + ECR image build |
| **Unified Governance (RBAC + Row Access Policies)** | Fine-grained access control on model predictions by role, team, or geography | Restrict access to PD scores by business unit or regulatory jurisdiction | Requires IAM policies per account/role + S3 bucket policies |
| **Incremental Feature Engineering** | Dynamic Tables only reprocess changed/new rows | 10x faster feature refresh for daily new applications vs full recompute | SageMaker Processing Step reprocesses entire dataset every run |

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 1 | `nb01_data_and_features.ipynb` | Environment setup (DEV/STAGING/PROD), synthetic data generation, Feature Store |
| 2 | `nb02_training_and_registry.ipynb` | XGBoost training, HPO, evaluation, registry, environment promotion |
| 3 | `nb03_serving_and_inference.ipynb` | SPCS endpoint deployment, latency benchmark, integration pattern |
| 4 | `nb04_monitoring_and_drift.ipynb` | ML Observability, drift detection, automated retraining alerts |
