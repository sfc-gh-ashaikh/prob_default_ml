# Probability of Default Demo: Presenter Guide

A narrative-driven talk track for a ~55-minute demo to a technical data science team. You are not walking through code line-by-line -- you are showing how Snowflake's platform features replace each component of their SageMaker architecture. Every section explains **what Snowflake feature is being used**, **what problem it solves**, and **how it compares to their current approach**.

Most cells are pre-run with cached outputs. Cells marked **RUN LIVE** are fast (<60s) and chosen for visual impact.

---

## Pre-Demo Setup (15-20 minutes before)

1. Open all 4 notebooks in Snowsight, run them end-to-end to cache outputs
2. Verify SPCS service is running: `SELECT SYSTEM$GET_SERVICE_STATUS('PD_DEMO_PROD.SERVING.PD_SCORING_SERVICE')` -- should return READY
3. Verify Model Monitor exists: `DESC MODEL MONITOR PD_DEMO_PROD.REGISTRY.PD_DRIFT_MONITOR`
4. Test the latency benchmark cell (nb03 cell 6) once to confirm the endpoint responds
5. Have the CUSTOMER_README open in a tab for architecture diagrams and cost tables

---

## Opening (5 minutes)

### The narrative:

"Thank you all for your time. What I want to do today is walk through your Probability of Default pipeline -- the one that currently runs on SageMaker -- and show you, step by step, how every component of that pipeline maps to a Snowflake capability. Not theoretically -- I have built the entire pipeline in working notebooks that we are going to run together.

Let me start with the big picture."

**Show the architecture diagrams from the CUSTOMER_README** (current SageMaker vs proposed Snowflake).

"On the left is what you have today. Your data starts in Snowflake, gets exported to S3, flows into SageMaker Pipelines where it goes through a Processing Step, HyperParameter Tuning Step, Training Step, Evaluation Step, and a Condition Step. If the model passes, it goes into the SageMaker Model Registry, and then Terraform deploys it to an endpoint. Monitoring is handled by SageMaker Model Monitor, which publishes to CloudWatch, which triggers EventBridge, which kicks off a retraining pipeline. That is 8 or more AWS services, all wired together with IAM policies and Terraform.

On the right is what we are proposing. Your data stays where it is -- in Snowflake. Everything else -- feature engineering, training, registry, deployment, monitoring, alerting -- runs on the same platform. The model is a Snowflake object. The endpoint is a Snowflake service. The monitor is a Snowflake object. No S3, no Terraform, no CloudWatch. One platform, one set of permissions, one identity.

Let me show you how."

---

## Notebook 1: Data & Feature Engineering (10 minutes)

### The narrative:

**Opening statement:**

"The first thing your pipeline does is prepare features. Today, that means a SageMaker Processing Step that pulls data out of Snowflake, stages it in S3, and runs a transformation job. Every time it runs, it reprocesses the entire dataset. In Snowflake, we replace that with something called a **Feature Store**, and the key difference is that the data never leaves the platform."

---

**Cells 2-3 (Environment setup + RBAC)** -- PRE-RUN, scroll through:

"Before we get to features, look at the infrastructure. These two cells set up three isolated environments -- DEV, STAGING, and PROD -- each as a separate Snowflake database with purpose-specific schemas. DEV has schemas for ML work, the Feature Store, and a model registry. STAGING has a registry and a serving schema for endpoint validation. PROD adds a monitoring schema for drift detection.

The second cell creates two RBAC roles. **PD_DS_DEV** gives your data science team full write access to DEV but read-only access to STAGING and PROD -- they can experiment freely but cannot deploy to production. **PD_MLOPS** has the opposite: full access to STAGING and PROD for model promotion and deployment.

This replaces what you currently manage with IAM policies across multiple AWS accounts and STS assume-role chains. In Snowflake, it is SQL GRANT statements, and it takes minutes instead of hours of Terraform configuration."

---

**Cell 5 (Data generation)** -- PRE-RUN, show the output:

"In production, your credit bureau data is already sitting in Snowflake -- you do not need to generate it. For this demo, I have created 500,000 synthetic credit applications that mirror what your pipeline would process. Each application has 20 features: 10 credit bureau features like credit score, outstanding balance, and utilisation ratio; 7 application features like debt-to-income ratio and applicant age; and an origination channel that gets one-hot encoded into 3 binary features. The default rate is about 6%, which is realistic for a BNPL credit risk portfolio.

The important point is: this data lives in Snowflake. In your current architecture, the first thing that happens is this data gets exported to S3 so SageMaker can access it. That data movement -- the S3 staging, the IAM permissions, the transfer costs -- is eliminated entirely."

---

**Cells 8-10 (EDA)** -- PRE-RUN, show the plots:

"Quick exploratory analysis. The histograms show how each of the 17 numeric features distributes differently between defaulters and non-defaulters -- you can see credit score is a strong separator. The correlation matrix on the bottom shows feature relationships. And this bar chart on the right shows default rates by origination channel -- Meta has a higher default rate than Direct. That channel-level difference is going to matter later when we set up segmented monitoring."

---

**Cells 12-14 (Feature Store)** -- PRE-RUN, explain the concepts:

"Now here is where it gets interesting. This is the **Snowflake Feature Store**. Let me explain what it does and why it matters.

A Feature Store solves three problems that your current Processing Step does not:

**First, incremental refresh.** When you run your SageMaker Processing Step, it reprocesses the entire dataset every time -- all 500,000 rows, even if only 10,000 new applications arrived since the last run. The Feature Store is backed by what Snowflake calls **Dynamic Tables**. A Dynamic Table only reprocesses the rows that changed. So when 10,000 new applications come in tomorrow, only those 10,000 rows are processed. The other 490,000 are untouched. That is a major cost and performance optimisation.

**Second, versioning.** We have created two Feature Views here -- one for the 17 numeric bureau features, one for the 3 origination channel features. Each is versioned (v1). When you want to add a new feature or change an encoding, you create v2. Both versions can run in parallel, so you can test v2 in DEV while v1 continues serving production. In SageMaker, changing your feature pipeline means rewriting the Processing Step and rerunning everything.

**Third -- and this is the big one for credit risk -- point-in-time correctness.**"

---

**Cell 16 (Training dataset)** -- **RUN LIVE**:

"Watch this. The `generate_training_set` call does something that is genuinely difficult to get right in SageMaker. It performs what is called an **ASOF JOIN** -- a temporal join that ensures each training example only uses features that were available at the time of the application.

Why does this matter? In credit risk modelling, if a customer's credit score was updated after they applied, and that updated score accidentally leaks into your training data, your model learns from information it would not have had at decision time. That is **data leakage**, and it can make your model look much better in development than it actually performs in production. Regulators specifically look for this.

In SageMaker, preventing data leakage is a manual exercise -- your team has to carefully manage timestamps in the Processing Step. Here, the Feature Store enforces it structurally. Every training set generated through this API is point-in-time correct by construction.

[Execute the cell] You can see the output: ~500,000 rows with 20 features plus the label. This is the dataset Notebook 2 will train on."

---

## Notebook 2: Training, Registry & Promotion (12 minutes)

### The narrative:

**Opening statement:**

"Now we train the model. I want to be clear about something: the XGBoost training code here is **identical** to what you run today. We are not asking you to learn a new framework or rewrite your model. What changes is what happens after training -- how the model gets versioned, validated, and promoted across environments."

---

**Cells 3-5 (Data split + Baseline)** -- PRE-RUN:

"Standard data science workflow -- 70/15/15 split with stratification to preserve the 6% default rate in each split. We train a baseline XGBoost with default hyperparameters to establish a performance floor. Nothing Snowflake-specific here -- this is the same sklearn and XGBoost code your team writes today."

---

**Cell 7 (HPO)** -- PRE-RUN:

"Bayesian hyperparameter optimisation with Optuna -- 30 trials searching across learning rate, tree depth, regularisation parameters. This replaces the SageMaker HyperParameter Tuning Step. The difference is operational: in SageMaker, HPO is a separate orchestrated job that spins up multiple training instances. Here, it runs in the notebook on warehouse compute. The warehouse auto-suspends after 60 seconds of inactivity between trials, so you only pay for active compute time.

For your scale, 30 trials on 500k rows runs in about 3-4 minutes. For larger-scale HPO, Snowflake also has a native Tuner API that can distribute trials across compute nodes -- but for XGBoost with 20 features, the in-notebook approach is fast enough."

---

**Cell 8 (Training cost)** -- PRE-RUN:

"Let me be transparent about training costs, because I know cost is a key part of this evaluation. Your current SageMaker pipeline runs 2.5 hours at about $1/hour -- roughly $2.50 per training run. That is genuinely cheap. A Snowflake MEDIUM warehouse costs about $12/hour, but because of auto-suspend, it is only active for about 15-20 minutes during training. That works out to roughly $4-5 per run -- about double.

So training is not where the savings come from. The savings come from three other places: the endpoint cost model, the services you no longer need to manage, and the engineering time you get back. We will see the full numbers in Notebook 3."

---

**Cells 11-12 (Evaluation + SHAP)** -- PRE-RUN for cell 11, **RUN LIVE** for cell 12:

"Full evaluation suite. Notice we report Gini coefficient and KS statistic alongside AUC -- these are standard metrics in credit risk that your team will expect to see. The calibration plot is particularly important: for a PD model, you do not just need high discrimination -- you need the predicted probabilities to be accurate, because those probabilities directly feed into your credit strategy and pricing.

[Execute SHAP cell] Now SHAP explainability. In regulated BNPL lending, when you decline an application, you are required to provide an adverse action notice explaining which factors contributed to the decision. These SHAP values give you exactly that -- feature-level explanations for each prediction. The bar chart shows global feature importance across the portfolio. The beeswarm plot on the right shows how each feature pushes individual predictions toward or away from default.

In SageMaker, you would set up a separate SageMaker Clarify job for this. Here, it is part of the training notebook."

---

**Cell 14-15 (Conditional registration)** -- **RUN LIVE**:

"Now here is where Snowflake's **Model Registry** comes in.

[Execute cell 14] First, there is a threshold gate -- we only register the model if AUC exceeds 0.70. This mirrors the SageMaker Condition Step in your pipeline. If the model is not good enough, it stops here.

When it passes, `log_model()` does several things in a single call:
- Registers the model as a **first-class Snowflake object** in the DEV registry
- Packages the XGBoost model for two target platforms: **WAREHOUSE** for SQL-native inference and **SNOWPARK_CONTAINER_SERVICES** for real-time REST endpoints
- Attaches all performance metrics as version metadata -- AUC, Gini, KS, Brier, Average Precision

In SageMaker, the equivalent requires the Condition Step, a CreateModel Step, and separate model packaging for the registry. Here it is one function call.

[Execute cell 15] Quick validation -- run 5 test rows through the registered model to confirm it works."

---

**Cells 17-20 (Promotion flow)** -- **RUN LIVE**:

"This is the part I am most excited to show you, because it replaces the most painful part of your current workflow.

In SageMaker, promoting a model from DEV to STAGING to PROD means: separate Terraform configurations per environment, ECR container image builds, IAM role assumptions, endpoint deployments. Each promotion can take 10-15 minutes and requires infrastructure-as-code expertise.

In Snowflake, watch:

[Execute cell 17] **Step 1: DEV to STAGING.** We load the model from the DEV registry, log it into the STAGING registry, and carry over all the metrics. That took a few seconds. In production, this step requires the PD_MLOPS role -- data scientists with PD_DS_DEV cannot perform this action.

[Execute cell 18] **Step 2: Validate in STAGING.** We run the full test set through the STAGING model and compare AUC against what we measured in DEV. The tolerance is 0.001. This catches any serialisation issues -- if the model behaves differently after promotion, it fails here. This is your integration testing gate.

[Execute cell 19] **Step 3: STAGING to PROD.** Same flow. Now the model is registered in the PROD registry, ready for endpoint deployment.

[Execute cell 20] And here is the complete picture -- one query shows the model registered across all three environments with its metrics at each stage. Full audit trail. Full lineage.

The entire promotion took about 2 minutes. In your current architecture, the same workflow takes hours and involves Terraform, IAM, and multiple AWS consoles."

---

## Notebook 3: Serving & Real-Time Inference (12 minutes)

### The narrative:

**Opening statement:**

"Your PD model serves a real-time endpoint that Taktile calls at the point of application -- when a customer applies for BNPL credit, the decisioning platform sends the application features to the model and gets back a probability of default. The target is around 50 milliseconds.

Today, that endpoint is deployed via Terraform to SageMaker, with an ECR container, an IAM role, an auto-scaling policy, and VPC configuration. Let me show you what it looks like in Snowflake."

---

**Cell 3 (Deploy)** -- PRE-RUN (pre-deployed):

"This is the deployment. One function call: `create_service()`. It takes the model from the PROD registry and deploys it as a live REST endpoint on what Snowflake calls **Snowpark Container Services** -- or SPCS. SPCS is a managed container runtime that runs inside Snowflake. The model is automatically containerised -- you do not build a Docker image, you do not push to ECR, you do not configure IAM roles.

The key parameters are the compute pool -- CPU_X64_S, which is a small CPU instance -- and the autoscaling bounds: minimum 1 instance, maximum 3. The service scales automatically under load without an Application Auto Scaling policy.

I pre-deployed this before our call because it takes about 5-10 minutes to initialise the first time. Let me confirm it is running."

---

**Cell 4 (Status check)** -- **RUN LIVE**:

"[Execute] Status is READY, with the ingress URL shown. That URL is what Taktile would call."

---

**Cells 6-7 (Latency benchmark)** -- **RUN LIVE**:

"[Execute cell 6] Now the moment of truth -- latency. I am sending 20 individual requests to the endpoint, each with all 20 model features, and measuring the round-trip time.

[Wait for results] Look at the numbers. Median latency is around [X]ms. P95 is [X]ms. P99 is [X]ms. For an XGBoost model with 20 features, this is well within your 50ms target for point-of-application decisioning.

[Execute cell 7] The histogram shows the distribution is tight -- no long-tail outliers. This is production-grade latency.

For comparison, SageMaker on an ml.m5.large typically achieves 50-100ms for a similar XGBoost payload. We are in the same range, sometimes faster."

---

**Cell 9 (Batch throughput)** -- PRE-RUN:

"In addition to real-time, Snowflake gives you batch inference without deploying an endpoint at all. Here we scored 1,000 records as a batch -- useful for strategy backtesting, portfolio analysis, and offline scoring. In SageMaker, batch scoring requires a separate Batch Transform job."

---

**Cell 11 (Taktile integration)** -- PRE-RUN:

"For Taktile integration, the pattern is identical to what they do with SageMaker today. Standard REST API, standard JSON payload with the 20 model features, standard JSON response with the PD score. The only two things that change are the endpoint URL and the authentication method -- you swap IAM/STS for a Snowflake PAT token. Same request format, same response format. Taktile could point to this endpoint in a day.

The setup steps are listed here: a network rule for access control, a network policy, a service role grant, and a PAT token. Four SQL statements replace the VPC endpoint configuration, IAM policies, and Terraform that you currently maintain."

---

**Cells 13-14 (Shadow testing)** -- **RUN LIVE**:

"This is a capability I want to highlight because it is something your team asked about in the architecture deck -- **shadow testing**.

In SageMaker, shadow testing means deploying a shadow variant on your endpoint via Terraform, configuring S3 to capture shadow predictions, and building a custom analysis pipeline to compare production versus candidate model outputs.

In Snowflake, watch this. [Execute cell 13] A single SQL query scores the same data through two model versions simultaneously -- V1 (production) and V2 (candidate). Both predictions come back in the same row, with the score delta calculated inline.

This is not an endpoint-based test -- it is a SQL operation that runs on warehouse compute. That means:
- You can compare any number of model versions at once, not just two
- It works retroactively on historical data, not just live traffic
- The comparison is a SQL query with GROUP BY for segment-level analysis
- You can embed it in a Dynamic Table for continuous shadow comparison

[Show cell 14] This cell shows the analysis pattern -- aggregate performance by version, then break it down by origination channel to catch hidden regressions.

SageMaker has no equivalent that is this simple."

---

**Cell 16 (SQL batch inference)** -- **RUN LIVE**:

"One more capability that SageMaker simply does not offer. [Execute] This is SQL-native inference -- calling the model directly in a SELECT statement using the `MODEL()!PREDICT_PROBA()` syntax. No endpoint, no Batch Transform job, no REST API.

Why does this matter? It means you can:
- Embed PD scoring directly in a Dynamic Table for continuous scoring as new applications arrive
- Run strategy backtesting as a SQL query
- Score any ad-hoc cohort in a worksheet
- Use it inside dbt models for downstream reporting

The model runs on your existing warehouse with auto-suspend. No additional infrastructure cost."

---

**Cell 18 (Cost comparison)** -- PRE-RUN:

"Let me walk through the cost comparison using your actual numbers, because I want to be honest about where Snowflake is cheaper and where it is not.

Your ml.m5.large endpoint costs $0.128/hour. Running 24/7, that is $92/month. The equivalent SPCS node -- CPU_X64_XS -- costs about $0.18/hour, or $130/month running 24/7. So for an always-on endpoint, SageMaker is actually about $38/month cheaper.

But here is where the economics change. First, your staging endpoint. In SageMaker, it costs the same $92/month, running 24/7 even though you only use it during model validation. In Snowflake, you spin up the staging service for validation and suspend it immediately after. That saves $87/month on its own.

Second, your production traffic is bursty -- BNPL applications peak during business hours and drop to near-zero overnight and on weekends. SPCS charges per second. If your service is active 8 hours/day, that is about $43/month versus SageMaker's $92.

Then add the services you no longer need: S3 staging ($10-50/month), CloudWatch and EventBridge ($10-30/month), the notebook instance ($34/month), and most importantly, the engineering time you spend on Terraform, IAM, and ECR -- conservatively 2-4 hours per deployment cycle.

The annual saving works out to roughly $4,300-7,400 per model pipeline. The CUSTOMER_README has the full breakdown with every number."

---

## Notebook 4: Monitoring & Drift Detection (10 minutes)

### The narrative:

**Opening statement:**

"The last piece of the pipeline, and the one that currently involves the most service-to-service wiring. In your architecture, SageMaker Model Monitor detects drift and publishes metrics to CloudWatch. A CloudWatch Alarm evaluates those metrics against a threshold. If the threshold is exceeded, an EventBridge rule triggers the SageMaker retraining pipeline. That is five services: Model Monitor, S3 for baseline data, CloudWatch for metrics, CloudWatch Alarms for thresholds, and EventBridge for orchestration. Each one has its own configuration surface and IAM requirements.

In Snowflake, the entire monitoring and retraining stack is native to the platform."

---

**Cells 3-6 (Inference log)** -- PRE-RUN:

"First, the data. I have loaded 600,000 inference predictions -- two months of production data at about 10,000 applications per day.

The first month is **normal** -- feature distributions match what the model was trained on. Credit scores centred around 650, utilisation ratio around 0.35, the same channel mix. The model is performing as expected.

The second month simulates a **macro-economic event** -- something like a downturn that changes the applicant population. Credit scores drop by about 70 points (from 650 to 580 average). Credit utilisation jumps from 35% to 50%. The channel mix shifts toward Meta, suggesting a marketing budget reallocation. Default rates rise from 6% to 12%.

This is the kind of drift that should trigger model retraining. Let me show you how Snowflake detects it."

---

**Cell 8 (Baseline)** -- PRE-RUN:

"The baseline table captures the first month's distributions -- the period where the model was performing well. This is the reference that the Model Monitor will compare against. In SageMaker, baseline data lives in S3 as JSON constraint files. Here, it is a Snowflake table."

---

**Cell 10 (Create Monitor)** -- **RUN LIVE**:

"[Execute] This is a **Model Monitor**. One SQL statement. Let me walk through what it does:

- `MODEL = PD_XGBOOST VERSION = 'V1'` -- it knows which model and version to monitor
- `SOURCE = INFERENCE_LOG` -- it reads predictions from this table
- `BASELINE = BASELINE_DATA` -- it compares against this reference distribution
- `REFRESH_INTERVAL = '1 day'` -- it recalculates metrics daily
- `PREDICTION_SCORE_COLUMNS = ('PREDICTED_PD')` -- it tracks the model's output
- `ACTUAL_SCORE_COLUMNS = ('ACTUAL_DEFAULT_90DPM')` -- it has access to ground truth for performance metrics
- `SEGMENT_COLUMNS = ('ORIGINATION_CHANNEL')` -- and here is the important one: it tracks everything **per origination channel**

That last point matters. If your Meta channel is degrading but Direct is stable, the overall metrics might look fine. Segmented monitoring catches channel-level problems that aggregate metrics miss. In SageMaker, segmented monitoring requires custom code per segment. Here, it is a parameter."

---

**Cell 11 (DESC)** -- **RUN LIVE**:

"[Execute] Confirm the configuration is correct."

---

**Cells 13-14 (Drift metrics)** -- **RUN LIVE**:

"[Execute cell 13] Now let me query the drift metrics. We are using **PSI -- Population Stability Index** -- which is the standard drift metric in credit risk. The thresholds are well established:
- PSI below 0.10: stable, no action needed
- PSI between 0.10 and 0.25: moderate drift, investigate
- PSI above 0.25: significant drift, retrain

[Show results] Look at the credit score -- PSI is well above 0.25. Credit utilisation ratio is also flagged. This is exactly what you would expect from the macro-economic scenario we simulated.

[Execute cell 14] The chart on the left shows PSI over time -- you can see exactly when the drift started. The chart on the right shows the latest PSI per feature with colour coding: green for stable, orange for moderate, red for alert. Credit score and utilisation are clearly in the red zone.

This is the same information that SageMaker Model Monitor would publish to CloudWatch -- but here, you query it directly with a SQL function. No CloudWatch dashboard configuration, no metric namespace setup."

---

**Cell 16 (Segmented performance)** -- PRE-RUN:

"Here is the segmented view I mentioned. AUC tracked per origination channel. If Meta traffic is degrading while Direct holds steady, your overall AUC might look acceptable, but your Meta credit strategy is mispriced. Catching this early prevents losses in a specific cohort that aggregate monitoring would miss."

---

**Cell 18 (Distribution plots)** -- PRE-RUN:

"Visual confirmation of the drift across 600,000 records. The overlaid histograms show the clear distribution shift between the normal period and the drifted period -- credit score shifted left, utilisation shifted right, applicant age shifted younger. These are the distribution changes driving the PSI alerts."

---

**Cell 20 (Alert + Task)** -- **RUN LIVE**:

"Now the automated response. [Execute]

This creates two objects:

A **Task** -- `PD_RETRAIN_TASK` -- which represents the retraining pipeline. In production, this calls a stored procedure that pulls new training data from the Feature Store, retrains the model with the same HPO configuration, evaluates it, and promotes it through DEV to STAGING to PROD. It is the equivalent of your SageMaker training pipeline.

An **Alert** -- `PD_DRIFT_ALERT` -- which checks PSI every 6 hours. The alert condition is a SQL query: 'does any feature have PSI above 0.25 in the last 7 days?' If yes, it fires the retrain Task.

In your current architecture, this same flow requires: SageMaker Model Monitor detecting drift, CloudWatch receiving the metrics, a CloudWatch Alarm evaluating the threshold, an EventBridge rule catching the alarm, and EventBridge triggering the SageMaker Pipeline. Five services, five configuration surfaces, five sets of IAM permissions.

In Snowflake: two objects, both defined in SQL, both running on your existing warehouse, both governed by the same RBAC roles."

---

**Cell 23 (Scheduled retrain)** -- **RUN LIVE**:

"[Execute] One more piece that was in your architecture deck -- scheduled retraining. In addition to drift-triggered retraining, you also want a fixed-cadence retrain, for example weekly, to incorporate new ground truth labels. This handles **gradual concept drift** that might not trigger PSI alerts but still degrades model calibration over time.

In SageMaker, this requires a separate EventBridge scheduled rule pointing to the same pipeline. In Snowflake, it is another Task with a CRON schedule. Both coexist: the drift-triggered Alert responds reactively to sudden shifts, and the scheduled Task runs proactively every week. Same warehouse, same RBAC, same query history audit trail."

---

**Cell 25 (Comparison table)** -- PRE-RUN:

"Here is the summary comparison. Nine capabilities, side by side. The key takeaway is not that any one of these is revolutionary on its own -- it is that all of them live on one platform, configured in one language, governed by one set of roles. The operational simplicity is the real value."

---

## Wrap-up & Q&A (5 minutes)

### The narrative:

"Let me bring it all together with five points.

**First, operational simplicity.** We replaced 8+ AWS services with a single platform. No Terraform for ML infrastructure. No cross-service IAM wiring. No CloudWatch alarm-to-EventBridge-to-Pipeline chains. Every object we created today -- the Feature Store, the model registry, the endpoint, the monitor, the alert, the scheduled task -- is a Snowflake object, managed with SQL and Python, governed by RBAC.

**Second, cost.** I have been honest about where the numbers fall. Your SageMaker training at $2.50 per run is cheap -- Snowflake is about $4-5. Where the savings come from: your staging endpoint runs 24/7 for $92/month when it only needs to run during validation -- that saves $87/month immediately. Your production endpoint for bursty BNPL traffic drops from $92 to about $43 with per-second billing. Add in the eliminated services and engineering time, and the annual saving is roughly $4,300-7,400 per model pipeline. The detailed breakdown is in the CUSTOMER_README.

**Third, performance.** We hit around [X]ms median latency on the real-time endpoint -- within your 50ms target for point-of-application decisioning. And SQL-native batch inference gives you strategy backtesting as a simple SQL query, which has no SageMaker equivalent.

**Fourth, governance.** For a regulated BNPL lender, this matters. DEV/STAGING/PROD with RBAC enforcement. Point-in-time correct feature engineering that prevents data leakage by design. SHAP explainability for adverse action notices. Model lineage and version metadata at every stage. Full audit trail in query history.

**Fifth, capabilities beyond parity.** Some things we showed today -- SQL-native scoring, shadow testing in a single query, segmented monitoring by origination channel, continuous scoring via Dynamic Tables, dual retraining triggers -- require significant custom engineering on SageMaker, if they are possible at all. On Snowflake, they are either built in or a few lines of SQL.

**The fundamental pitch is this:** your data is already in Snowflake. Today, you export it to S3, process it in SageMaker, and bring the results back. We are proposing to bring the compute to the data instead of moving the data to the compute. The XGBoost code does not change. What changes is the infrastructure around it -- and that infrastructure gets dramatically simpler."

---

## Anticipated Questions & Answers

**"What is the migration timeline?"**
About 2 weeks for a focused team. The model code is identical XGBoost -- what changes is the orchestration layer. The notebooks we showed today are essentially the migration template.

**"Can Taktile switch that easily?"**
Yes. Same REST API, same JSON format. The only change is the endpoint URL and the authentication method (PAT instead of IAM). Taktile could point to the Snowflake endpoint in a day.

**"What about GPU for future models?"**
SPCS supports GPU compute pools (A10G, A100). For XGBoost you do not need GPU, but when you move to deep learning models in the future, the same platform supports it.

**"How does this work with multiple AWS accounts / multi-account strategy?"**
Snowflake supports database replication across accounts and regions. You can replicate the PROD registry to a DR account, for example. The same RBAC model applies.

**"What about CI/CD?"**
The `snow` CLI integrates with GitHub Actions, GitLab CI, or any CI/CD system. Model promotion can be scripted as `snow model` commands in a pipeline.

**"PSI is limited -- what about other drift metrics?"**
Model Monitor also supports KL divergence. For credit risk, PSI is the industry standard that regulators expect to see. But you can write any metric as the Alert condition -- it is a SQL query.

**"Can we set different thresholds per feature?"**
Yes. The Alert condition is a SQL query, so you can write logic like: 'if credit_score PSI > 0.20 OR credit_utilisation PSI > 0.30 OR any feature PSI > 0.50'. Per-feature thresholds, composite scores, whatever your risk policy requires.

**"What about the Feature Store vs plain views or tables?"**
For a static dataset, a plain table would work. The Feature Store adds three things you need in production: incremental refresh (only new rows processed), point-in-time correctness (ASOF JOIN for training sets), and versioning (run v1 and v2 in parallel). These are the features that prevent data leakage and reduce cost at scale.

**"How does auto-suspend work for the endpoint?"**
SPCS services with public endpoints do not auto-suspend on their own -- you need to explicitly suspend the service via a scheduled Task or manual command, then it auto-resumes when traffic arrives. For production with min_instances=1, the service stays warm. For staging, you suspend after validation and only pay for the minutes you used.

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
