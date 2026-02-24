# SecureGuard Fraud Firewall

An insurance claim fraud detection system that scores incoming claims in real time, flags high-risk cases for investigation, and auto-approves routine ones — built on a Random Forest classifier with advanced feature engineering and class-balancing.

---

## How It Works

The pipeline has five stages:

**1. Feature Engineering**
Three new features are created before any model sees the data:
- `Age_Is_Missing` — a binary flag set to `1` when `Age == 0`, catching a known data-quality anomaly that correlates with fraud.
- `Customer_Profile_Cluster` — a KMeans (k=5) cluster label grouping customers by behavioural similarity across `Age`, `Deductible`, and `DriverRating`. Unsupervised learning gives the classifier a pre-computed "what type of customer is this?" signal.
- `Price_Policy_Interaction` — a concatenation of `VehiclePrice` and `BasePolicy`, then frequency-encoded, capturing the base rate of fraud for each vehicle-price/policy-type combination.

**2. Encoding**
High-cardinality categorical columns (`Make`, `Price_Policy_Interaction`) are frequency-encoded rather than one-hot or label-encoded, preserving ordinal fraud-rate information without dimensionality explosion. All remaining categoricals are label-encoded.

**3. Class Balancing**
The training set is rebalanced to a **1:3 fraud-to-legitimate ratio** via downsampling of the majority class. This is more aggressive than keeping the natural imbalance (~6% fraud) but less extreme than 1:1 SMOTE, tuned to keep recall high without destroying precision entirely.

**4. Model**
A `RandomForestClassifier` (100 trees, max depth 10) is trained on the balanced set. The variable is named `stacking_clf` — a placeholder for a planned `StackingClassifier` upgrade (imports are already in place).

**5. Scoring & Triage**
`secureguard_mvp_system()` converts raw model probabilities into a 0–100 risk score, applies a **25-point threshold** to assign one of two actions, and generates a `Red_Flags` field explaining why the score is elevated.

---

## Threshold Logic

| Risk Score | Action | Meaning |
|---|---|---|
| ≥ 25 | 🚨 INVESTIGATE | Routed to a human investigator |
| < 25 | ✅ AUTO-APPROVE | Processed automatically |

The threshold is set deliberately low (25 out of 100) to maximise recall — catching more fraud at the cost of some false positives. Adjust based on your operational cost of a missed fraud vs. a wrongly-flagged claim.

---

## Red Flag Rules

The current flag logic is rule-based on top of the model score:

| Condition | Flag |
|---|---|
| `Fault == 0` | Policy Holder at Fault |
| `Age_Is_Missing == 1` | Missing Age Data |
| `Deductible > 400` | High Deductible Anomaly |
| None of the above | Routine Claim |

---

## Project Structure

```
.
├── fraud_oracle.csv              # Input dataset (not included — add your own)
├── secureguard.py                # Main pipeline script
└── SecureGuard_MVP_Dashboard.csv # Output: scored claims (auto-generated on run)
```

---

## Requirements

```
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install pandas numpy scikit-learn
```

---

## Usage

1. Place `fraud_oracle.csv` in the project root. The file must contain a `PolicyNumber` column (dropped automatically) and a `FraudFound_P` target column.

2. Run the script:

```bash
python secureguard.py
```

3. The console will print the top 5 highest-risk claims from the test batch. The full scored output is written to `SecureGuard_MVP_Dashboard.csv`.

**Output columns:**

| Column | Description |
|---|---|
| `Risk_Score` | 0–100 fraud probability score |
| `Action` | INVESTIGATE or AUTO-APPROVE |
| `Red_Flags` | Human-readable reasons for the score |

---

## Known Limitations & Next Steps

- **Threshold is hardcoded at 25** — should be tuned against a precision-recall curve on a held-out validation set before production use.
- **Red flag logic is rule-based** — planned upgrade is SHAP values per prediction so every flag is model-derived, not hand-written.
- **`stacking_clf` is currently a single RandomForest** — the `StackingClassifier` / `VotingClassifier` imports are already included; the ensemble upgrade is the logical next step.
- **Label encoding uses a single `LabelEncoder` instance** across all columns — this means the encoder state is overwritten on each loop iteration and cannot be reused for inference on new data. Fit a separate encoder per column and persist them alongside the model for production.
- **No model persistence** — add `joblib.dump(model, 'secureguard_model.pkl')` after training so the model doesn't need to be retrained on every run.
