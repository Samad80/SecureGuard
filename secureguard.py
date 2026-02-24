import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, average_precision_score
from sklearn.utils import resample
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# 1. Load Data
df = pd.read_csv('fraud_oracle.csv')
df_clean = df.drop(['PolicyNumber'], axis=1)

# 2. Advanced Feature Engineering
# A. The "Anomaly" Flag
df_clean['Age_Is_Missing'] = (df_clean['Age'] == 0).astype(int)

# B. Unsupervised Learning Feature: "Customer Profile Cluster"
# Groups customers by behavior (Age, Deductible, DriverRating)
cluster_cols = ['Age', 'Deductible', 'DriverRating']
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_clean['Customer_Profile_Cluster'] = kmeans.fit_predict(df_clean[cluster_cols])

# C. Interaction Feature: "High Value & Risky"
# Combining Vehicle Price and Policy Type for a richer signal
df_clean['Price_Policy_Interaction'] = df_clean['VehiclePrice'] + "_" + df_clean['BasePolicy']

# 3. Frequency Encoding (The Pro Move)
for col in ['Make', 'Price_Policy_Interaction']:
    freq = df_clean[col].value_counts(normalize=True)
    df_clean[f'{col}_Freq'] = df_clean[col].map(freq)

# Label Encoding for the rest
le = LabelEncoder()
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = le.fit_transform(df_clean[col])

# 4. Split & Balance (1:3 Ratio)
X = df_clean.drop('FraudFound_P', axis=1)
y = df_clean['FraudFound_P']
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = pd.concat([X_train_full, y_train_full], axis=1)
fraud = train_data[train_data.FraudFound_P == 1]
legit = train_data[train_data.FraudFound_P == 0]

legit_downsampled = resample(legit, replace=False, n_samples=len(fraud)*3, random_state=42)
df_train_balanced = pd.concat([legit_downsampled, fraud])

X_train = df_train_balanced.drop('FraudFound_P', axis=1)
y_train = df_train_balanced['FraudFound_P']

# 5. The Ensemble Model
stacking_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
stacking_clf.fit(X_train, y_train)

def secureguard_mvp_system(new_claims_df, model):
    """
    The Core MVP Engine: Processes claims and outputs an actionable dashboard.
    """
    # 1. Generate Risk Scores (0-100)
    probabilities = model.predict_proba(new_claims_df)[:, 1]
    scores = (probabilities * 100).round(2)
    
    # 2. Build the Dashboard View
    dashboard = new_claims_df.copy()
    dashboard['Risk_Score'] = scores
    
    # 3. Categorize (Threshold set at 25 for high sensitivity)
    dashboard['Action'] = dashboard['Risk_Score'].apply(
        lambda x: '🚨 INVESTIGATE' if x >= 25 else '✅ AUTO-APPROVE'
    )
    
    # 4. Extract 'Red Flags' (Example logic based on feature importance)
    # This simulates an AI 'commenting' on the claim
    flags = []
    for i, row in dashboard.iterrows():
        reasons = []
        if row['Fault'] == 0: reasons.append("Policy Holder at Fault") # Assuming 0 is PH
        if row['Age_Is_Missing'] == 1: reasons.append("Missing Age Data")
        if row['Deductible'] > 400: reasons.append("High Deductible Anomaly")
        flags.append(", ".join(reasons) if reasons else "Routine Claim")
    
    dashboard['Red_Flags'] = flags
    
    # Return only high priority cases for the demo
    return dashboard.sort_values(by='Risk_Score', ascending=False)

# --- DEMO EXECUTION ---
# Simulate 50 new claims arriving at the SecureGuard portal
test_batch = X_test.head(50) 
mvp_output = secureguard_mvp_system(test_batch, stacking_clf)

# Display the top 5 'Most Dangerous' claims for the judges
print("=== SECUREGUARD FRAUD FIREWALL: LIVE TRIAGE ===")
print(mvp_output[['Risk_Score', 'Action', 'Red_Flags']].head(5))
print("\nFull MVP Output Sample:")
print(mvp_output.head(10))


# Export to CSV for the 'Investigator Dashboard' demo
mvp_output.to_csv('SecureGuard_MVP_Dashboard.csv', index=False)