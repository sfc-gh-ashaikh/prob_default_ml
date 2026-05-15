# Probability of Default Demo: Talk Track

Presenter notes for a ~55-minute notebook-based demo to a technical data science team.

Most cells are **pre-run** with outputs cached. Cells marked **RUN LIVE** are executed during the demo for maximum impact -- they are all fast (<60s) and demonstrate key Snowflake differentiators.

---

## Pre-Demo Setup Checklist (15-20 minutes before)

Run these steps before the meeting starts:

1. **Run all notebooks end-to-end** to populate cell outputs:
   ```
   Open each notebook in Snowsight and Run All, or execute locally.
   ```
2. **Verify SPCS service is running** (this is the 5-10 min bottleneck):
   ```sql
   SELECT SYSTEM$GET_SERVICE_STATUS('PD_DEMO_PROD.SERVING.PD_SCORING_SERVICE');
   -- Should show: READY, min_instances=1
   ```
   If not running, execute the `create_service()` cell in nb03 and wait for READY status.
3. **Verify Model Monitor exists**:
   ```sql
   DESC MODEL MONITOR PD_DEMO_PROD.REGISTRY.PD_DRIFT_MONITOR;
   ```
4. **Open all 4 notebooks** in separate Snowsight tabs so you can switch quickly.
5. **Open the CUSTOMER_README** in a tab for the architecture diagrams.
6. **Test the latency benchmark cell** (nb03 cell 6) once to confirm the endpoint responds.

---

## Opening (5 minutes)

### What to say:
"Today we are going to walk through how a BNPL fintech can run their entire Probability of Default ML pipeline on Snowflake -- from feature engineering through to production serving and monitoring. Your data already lives in Snowflake. The question is: why move it to SageMaker and back?"

### What to show:
- The architecture comparison diagram from the CUSTOMER_README (current vs proposed)
- The component mapping table (12 rows, now includes shadow testing + scheduled retraining)
- The environment strategy (DEV/STAGING/PROD)

### Key points:
- Data already in Snowflake -- why move it to S3/SageMaker?
- Current architecture: 8+ services to manage. Proposed: 1 platform.
- No Terraform for ML infrastructure.
- Same XGBoost code -- this is not a rewrite, it is a relocation.

---

## Notebook 1: Data & Feature Engineering (10 minutes)

### What to say:
"Let me start by showing how we set up the infrastructure. In SageMaker, this means Terraform configurations, AWS Organizations, IAM roles, VPC networking. In Snowflake, it is 20 lines of SQL."

### Cell-by-cell talking points:

**Cell 2 (Environment setup)** -- PRE-RUN: "Three databases -- DEV, STAGING, PROD -- created in seconds. Each has purpose-specific schemas. No Terraform plan, no apply, no state file." Show the output.

**Cell 3 (RBAC roles)** -- PRE-RUN: "RBAC roles control who can do what in each environment. Your DS team gets full access to DEV and read-only access to STAGING/PROD. ML Ops can promote models. This replaces IAM policies across multiple AWS accounts."

**Cell 5 (Data generation)** -- PRE-RUN: "In production, your credit bureau data is already in Snowflake. We generate 500k synthetic credit applications here. Notice realistic distributions -- credit score centred around 650, ~6% default rate, three origination channels."

**Cell 6 (Row counts)** -- PRE-RUN: Show the 500k total with default rate breakdown.

**Cells 8-10 (EDA + plots)** -- PRE-RUN: "Quick data quality check. The box plots show feature distributions split by default status. Default rate varies by origination channel -- we will use this for segmented monitoring in Notebook 4."

**Cells 12-14 (Feature Store)** -- PRE-RUN: "The Feature Store replaces your SageMaker Processing Step and S3 staging. Features are backed by Dynamic Tables -- incremental refresh, only changed rows are reprocessed. Two feature views: 17 bureau features and 3 origination channel features."

**Cell 16 (Training dataset)** -- **RUN LIVE**: "This is the killer feature for credit risk. Watch: `generate_training_set` uses ASOF JOIN to ensure point-in-time correctness. No future data leaks into your training set. In SageMaker, you handle this manually -- and it is error-prone. Here it is enforced by the platform." (~45s)

**Cell 17 (Save)** -- PRE-RUN: Show the saved table confirmation.

### Objection handling:
- **"Can Dynamic Tables handle our data volume?"** -- Dynamic Tables support incremental refresh on billions of rows. The Feature Store is backed by Snowflake's scalable compute.
- **"What about feature versioning?"** -- Feature views are versioned. You can run multiple versions in parallel.

---

## Notebook 2: Training, Registry & Promotion (12 minutes)

### What to say:
"Now we train the model. The key message here: this is the same XGBoost code you write today. Nothing changes in your training logic. What changes is how you manage, version, and promote models across environments."

### Cell-by-cell talking points:

**Cell 3 (Data split)** -- PRE-RUN: "Standard 70/15/15 split with stratification. Class weights calculated for the ~6% default rate."

**Cell 5 (Baseline)** -- PRE-RUN: "Quick baseline with default hyperparameters. AUC of ~X. We will improve on this with HPO."

**Cell 7 (HPO)** -- PRE-RUN: "Bayesian optimization with Optuna -- 30 trials. This replaces the SageMaker HyperParameter Tuning Step. Same algorithm, no SageMaker orchestration overhead. Best trial improved AUC by X points. For larger-scale HPO, Snowflake also has a native Tuner API that distributes across compute nodes."

**Cell 8 (Training cost comparison)** -- PRE-RUN: "Let me be upfront about training costs. Your current SageMaker pipeline runs 2.5 hours at $1/hour -- about $2.50 per training run. Snowflake warehouse compute costs more per hour -- a MEDIUM warehouse is about $12/hour -- but runs for a shorter active window due to auto-suspend. Roughly $4-5 per run. Training compute cost is comparable, maybe slightly higher. But the savings are not in training -- they are in the endpoint, the eliminated services, and the engineering time. We will see the full picture in Notebook 3."

**Cell 9 (Final model)** -- PRE-RUN: Show the trained model output.

**Cell 11 (Evaluation plots)** -- PRE-RUN: "Full evaluation suite. Notice we report Gini and KS alongside AUC -- these are standard in credit risk. The calibration plot is critical: PD models need well-calibrated probabilities, not just high AUC."

**Cell 12 (SHAP)** -- **RUN LIVE**: "Built-in SHAP explainability. In a regulated BNPL environment, you need feature-level explanations for adverse action notices." (~30s, visually impressive waterfall plot)

**Cell 14 (Conditional registration)** -- **RUN LIVE**: "This is the SageMaker Condition Step equivalent. We only register if AUC exceeds the threshold. Watch: `log_model()` -- the model goes into the DEV registry with all metrics attached as version metadata. One Python call replaces Terraform + ECR + IAM." (~60s)

**Cell 15 (Validation inference)** -- **RUN LIVE**: "Quick proof the registered model works -- call `mv.run()` on 5 test rows." (~5s)

**Cells 17-20 (DEV -> STAGING -> PROD promotion)** -- **RUN LIVE**: "THIS is the workflow that replaces Terraform. Watch each step:
- Cell 17: Load from DEV, log into STAGING with metrics carried over.
- Cell 18: Validate STAGING -- confirm metrics match, run inference.
- Cell 19: Promote to PROD -- same flow, RBAC-governed.
- Cell 20: One query shows the model registered across all three environments.

In SageMaker, this same workflow requires separate Terraform configs per environment, manual approval in the SM Registry, and IAM role assumption chains. Here it is 4 Python calls." (~2 min total)

### Objection handling:
- **"What about IaC for model deployment?"** -- Model objects are schema-level Snowflake objects. They can be managed via the `snow` CLI in CI/CD. No Terraform needed for ML-specific infrastructure.
- **"How do you enforce approval gates?"** -- RBAC. The PD_DS_DEV role cannot write to STAGING or PROD. Only PD_MLOPS can promote.

---

## Notebook 3: Serving & Real-Time Inference (12 minutes)

### What to say:
"Now let me show you something that took a team a week with Terraform -- deploying a production endpoint. In Snowflake, it is one function call."

### Cell-by-cell talking points:

**Cell 3 (Deploy)** -- PRE-RUN (pre-deployed before demo): "One call: `create_service()`. No Terraform plan, no ECR image build, no IAM role, no VPC endpoint configuration. The model is deployed from the PROD registry to SPCS with autoscaling. I pre-deployed this 15 minutes ago -- let me verify it is running."

**Cell 4 (Status check)** -- **RUN LIVE**: "Service status: READY. Running with min_instances=1, autoscaling up to 3." (~2s)

**Cells 6-7 (Latency benchmark)** -- **RUN LIVE**: "[Execute cell 6] Let me hit the endpoint 20 times and measure latency. Your target is 50ms for point-of-application decisioning. [Show results] Median latency of ~Xms. [Execute cell 7] The histogram confirms tight distribution around the median. P95 and P99 are acceptable for real-time decisioning. This is XGBoost on CPU -- a lightweight model with 20 features." (~30s)

**Cell 9 (Batch throughput)** -- PRE-RUN: "For offline scoring and strategy backtesting, batch inference gives you high throughput without needing the endpoint at all."

**Cell 11 (Taktile integration)** -- PRE-RUN: "The integration pattern is identical to what Taktile does with SageMaker today. Standard REST API with JSON payload. The only change is the endpoint URL and authentication method -- PAT instead of IAM. Same request/response format. They can point to Snowflake in under a day."

**Cells 13-14 (Shadow testing)** -- **RUN LIVE**: "This is something SageMaker requires a separate Shadow Testing setup for -- Terraform config, S3 capture, custom analysis pipeline. In Snowflake, watch: [Execute cell 13] We score the same data with two model versions in a single SQL query. Production V1 and candidate V2 side by side, with score delta calculated inline. [Show cell 14] The aggregate comparison pattern shows you mean, stddev, and segment-level differences. You can compare N versions simultaneously, and it works retroactively on historical data -- not just live traffic." (~10s)

**Cell 16 (SQL batch inference)** -- **RUN LIVE**: "This is a capability SageMaker simply does not have. `MODEL()!PREDICT_PROBA()` directly in SQL -- no endpoint needed. Embed PD scoring in Dynamic Tables, dbt models, or ad-hoc queries. Strategy backtesting becomes a SQL query." (~5s)

**Cell 18 (Cost comparison)** -- PRE-RUN: "Let me use your actual numbers. Your ml.m5.large endpoint costs $0.128/hour -- that is $92/month per endpoint running 24/7. Snowflake SPCS on the smallest CPU node, CPU_X64_XS, is 0.06 credits/hour -- about $0.18/hour at Enterprise pricing. Running 24/7, that is $130/month -- so for always-on, SageMaker is actually cheaper by about $38. But here is the difference: your staging endpoint also costs $92/month in SageMaker, running 24/7. In Snowflake, you only spin it up during validation -- effectively $5/month. That is an $87 saving right there. And for prod, BNPL traffic is bursty -- peak hours only. At 8 hours/day, SPCS drops to about $43/month versus SageMaker's $92. On training, your $2.50 per run is genuinely cheap -- Snowflake is about $4-5. Training cost is comparable, maybe slightly higher. But add up the full stack -- staging endpoint, S3, CloudWatch, EventBridge, notebook instance, engineering time -- and the annual saving is roughly $4,300-7,400 per model pipeline."

### Objection handling:
- **"What about cold start?"** -- SPCS services can be configured with min_instances=1 to stay warm. Auto-suspend after configurable idle timeout.
- **"What if we need GPU?"** -- SPCS supports GPU compute pools (A10G, A100). For XGBoost you do not need GPU, but for future deep learning models it is there.
- **"Can Taktile really switch that easily?"** -- Yes. Same REST pattern, same JSON format. Only the URL and auth method change.
- **"How does shadow testing handle live traffic?"** -- For real-time shadow testing, you can deploy two SPCS services and use a lightweight proxy or the Taktile platform to fan out requests. For batch comparison (shown here), SQL scoring is instant and retroactive.

---

## Notebook 4: Monitoring & Drift Detection (10 minutes)

### What to say:
"The last piece. In your current architecture, you have SageMaker Model Monitor publishing to CloudWatch, CloudWatch Alarms triggering EventBridge, EventBridge kicking off SageMaker Pipelines. That is five services to configure and wire together. Let me show you how Snowflake does it."

### Cell-by-cell talking points:

**Cells 3-6 (Inference log data)** -- PRE-RUN: "We loaded 600k inference records -- two months of data at ~10k/day. The first month matches training distributions. The second month simulates a macro-economic event -- credit scores drop, utilisation goes up, channel mix shifts toward Meta. This is realistic drift for a BNPL lender."

**Cell 8 (Baseline table)** -- PRE-RUN: "Baseline from training data -- the reference distribution for drift detection."

**Cell 10 (Create Monitor)** -- **RUN LIVE**: "One SQL statement. That is it. [Execute] This replaces: the SageMaker Model Monitor configuration, the CloudWatch metric namespace setup, and the baseline constraint JSON. Notice the SEGMENT_COLUMNS -- we segment by origination channel so we can catch drift in specific cohorts that overall metrics would miss." (~10s)

**Cell 11 (DESC Monitor)** -- **RUN LIVE**: "Verify the monitor is configured correctly." (~2s)

**Cells 13-14 (Drift metrics + plot)** -- **RUN LIVE**: "PSI per feature over time. [Execute] Look at the credit score -- PSI well above 0.25, confirming significant drift. Credit utilisation ratio is also drifted. This is exactly what you would see if a recession hit your customer base. The chart shows the progression day by day." (~10s)

**Cell 16 (Segmented performance)** -- PRE-RUN: "AUC by origination channel. If Meta traffic degrades while Direct holds steady, overall AUC might look fine but your Meta credit strategy is mispriced. This segmented view is built into Model Monitor -- in SageMaker you write custom analysis code for this."

**Cell 18 (Distribution plots)** -- PRE-RUN: "Visual proof of the drift we injected. Clear distribution shift in credit score, outstanding balance, and utilisation ratio."

**Cell 20 (Drift Alert + Task)** -- **RUN LIVE**: "The drift-triggered retraining pipeline. [Execute] Two SQL objects: an Alert that checks PSI every 6 hours, and a Task it fires when the threshold is exceeded. In production, this Task calls a stored procedure that retrains in DEV, validates in STAGING, promotes to PROD. The entire CloudWatch + EventBridge + SageMaker Pipelines stack -- replaced by two Snowflake objects." (~5s)

**Cell 21 (Architecture comparison print)** -- PRE-RUN: Walk through the 5-service vs 2-object comparison.

**Cell 23 (Scheduled retrain)** -- **RUN LIVE**: "In addition to drift-triggered retraining, you also want a fixed-cadence retrain -- weekly or monthly -- to incorporate new ground truth labels. In SageMaker, this requires a separate EventBridge scheduled rule. In Snowflake: one more Task with a CRON schedule. [Execute] Both mechanisms coexist: drift-triggered for reactive response, scheduled for proactive freshness. Same warehouse, same RBAC, same audit trail." (~5s)

**Cell 25 (Comparison table)** -- PRE-RUN: Walk through the capability-by-capability comparison.

### Objection handling:
- **"PSI is limited. What about KL divergence, KS test?"** -- KL divergence is also supported. For credit risk, PSI is the industry standard.
- **"Can we set different thresholds per feature?"** -- The Alert condition is a SQL query. You can write any logic you want -- per-feature thresholds, composite scores, etc.
- **"What about performance monitoring, not just drift?"** -- Model Monitor tracks AUC, F1, precision, recall, and custom metrics. We showed segmented AUC by channel.

---

## Wrap-up & Q&A (5 minutes)

### What to say:
"Let me summarise what we have shown:

1. **Operational simplicity**: We replaced 8+ AWS services with a single Snowflake platform. No Terraform, no cross-service IAM, no CloudWatch wiring.

2. **Cost**: Let me give you the honest numbers. Your SageMaker training at $2.50 per run is cheap -- Snowflake is comparable at $4-5 per run. Where the savings come from: your staging endpoint runs 24/7 for $92/month when it only needs to run during validation -- that alone saves $87/month on Snowflake. Your prod endpoint at $0.128/hour is actually cheaper than SPCS for always-on, but for bursty BNPL traffic at 8 hours/day, SPCS drops to about $43 versus $92. Add in eliminated S3 staging, CloudWatch, EventBridge, notebook instances, and engineering time -- the total saving is roughly $4,300-7,400 per year per model pipeline. The CUSTOMER_README has the full breakdown with every number sourced.

3. **Performance**: XGBoost inference at ~50ms -- meeting your target for point-of-application decisioning. SQL-native batch inference for strategy backtesting.

4. **Governance**: DEV/STAGING/PROD with RBAC. Model lineage. Point-in-time correct feature engineering. SHAP explainability for regulatory compliance.

5. **Beyond parity**: Shadow testing via SQL, continuous scoring with Dynamic Tables, scheduled + drift-triggered retraining, segmented monitoring -- capabilities that require significant custom engineering on SageMaker.

6. **Migration effort**: Approximately 2 weeks for a focused team. The XGBoost code is identical -- what changes is the orchestration layer around it.

The data is already in Snowflake. We are proposing to bring the compute to the data instead of moving the data to the compute."

### Questions to anticipate:
- Migration timeline and effort (2 weeks for focused team)
- Taktile switching cost (minimal -- same REST pattern)
- GPU support for future models (SPCS supports A10G, A100)
- Multi-account vs single-account (database replication available)
- CI/CD integration (snow CLI + GitHub Actions)
- Regulatory requirements (SHAP, audit trail, RBAC, model lineage)
- Shadow testing for live traffic (SPCS multi-service + proxy, or batch SQL comparison)
- Scheduled vs drift-triggered retraining cadence (both supported, configurable)

---

## Timing Summary

| Section | Duration | Live cells | Pre-run cells |
|---|---|---|---|
| Opening | 5 min | 0 | 0 |
| NB1: Data & Features | 10 min | 1 (training dataset) | 10 |
| NB2: Training & Registry | 12 min | 6 (SHAP, registration, validation, 3x promotion) | 8 |
| NB3: Serving & Inference | 12 min | 5 (status, latency x2, shadow testing, SQL inference) | 5 |
| NB4: Monitoring & Drift | 10 min | 5 (monitor, desc, drift metrics, alert+task, scheduled retrain) | 8 |
| Wrap-up & Q&A | 5 min | 0 | 0 |
| **Total** | **~54 min** | **17 cells** | **31 cells** |

---

## Post-Demo Follow-up

Suggested next steps to propose:
1. **Proof of concept**: Port one existing PD model to Snowflake (2-week sprint)
2. **Taktile integration test**: Point Taktile at a Snowflake SPCS endpoint in staging
3. **Cost analysis**: Compare 3 months of SageMaker costs vs projected Snowflake costs
4. **DS team workshop**: Hands-on session with the Risk DS team using these notebooks
