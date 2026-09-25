# ================================================================
# LoanSense — AI-Powered Credit Risk Explainer
# Prototype explainability pipeline for credit decisions,
# designed around EU AI Act transparency requirements
#
# Run this in Google Colab — all cells in order
# ================================================================

# ── CELL 1: Install dependencies ─────────────────────────────
# !pip install shap langchain langchain-openai openai pandas scikit-learn matplotlib

# ── CELL 2: Imports ──────────────────────────────────────────
import hashlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("✓ All imports successful")

# ── CELL 3: Load German Credit Dataset ───────────────────────
# UCI Statlog German Credit Dataset
# 1000 applicants, 20 features, binary target (1=Good, 2=Bad)
# Note: monetary values are in Deutsche Mark (DM), not euros.

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

columns = [
    'checking_account', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'employment', 'installment_rate',
    'personal_status', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installments', 'housing', 'existing_credits',
    'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'target'
]

df = pd.read_csv(url, sep=' ', header=None, names=columns)

# Convert target: 1=Good → 0, 2=Bad → 1
df['target'] = df['target'].map({1: 0, 2: 1})

print(f"✓ Dataset loaded: {df.shape[0]} applicants, {df.shape[1]-1} features")
print(f"  Good risk: {(df['target']==0).sum()} | Bad risk: {(df['target']==1).sum()}")

# ── CELL 4: Feature Engineering ──────────────────────────────
# Categorical columns hold UCI codes like 'A11'. LabelEncoder sorts them
# alphabetically, so the integer codes map as documented in CODEBOOK below.
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include='object').columns

le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Human-readable feature names for explanation
feature_names = [
    'Checking Account Status', 'Loan Duration (months)', 'Credit History',
    'Loan Purpose', 'Loan Amount (DM)', 'Savings Account',
    'Employment Duration', 'Installment Rate (1-4)', 'Personal Status & Sex',
    'Other Debtors', 'Residence Duration (1-4)', 'Property Type',
    'Age', 'Other Installment Plans', 'Housing Type',
    'Existing Credits', 'Job Type', 'Number of Dependents',
    'Has Telephone', 'Foreign Worker'
]

# Integer code → meaning, after alphabetical label encoding
CODEBOOK = {
    'Checking Account Status': {0: '< 0 DM', 1: '0–200 DM', 2: '≥ 200 DM', 3: 'no checking account'},
    'Credit History': {0: 'no credits / all paid', 1: 'all paid at this bank', 2: 'existing credits paid duly',
                       3: 'past payment delays', 4: 'critical account / credits elsewhere'},
    'Loan Purpose': {0: 'new car', 1: 'used car', 2: 'other', 3: 'furniture', 4: 'radio/TV',
                     5: 'appliances', 6: 'repairs', 7: 'education', 8: 'retraining', 9: 'business'},
    'Savings Account': {0: '< 100 DM', 1: '100–500 DM', 2: '500–1000 DM', 3: '≥ 1000 DM', 4: 'unknown / none'},
    'Employment Duration': {0: 'unemployed', 1: '< 1 year', 2: '1–4 years', 3: '4–7 years', 4: '≥ 7 years'},
    'Personal Status & Sex': {0: 'male, divorced/separated', 1: 'female, divorced/separated/married',
                              2: 'male, single', 3: 'male, married/widowed'},
    'Other Debtors': {0: 'none', 1: 'co-applicant', 2: 'guarantor'},
    'Property Type': {0: 'real estate', 1: 'savings agreement / life insurance', 2: 'car or other', 3: 'unknown / none'},
    'Other Installment Plans': {0: 'bank', 1: 'stores', 2: 'none'},
    'Housing Type': {0: 'rent', 1: 'own', 2: 'for free'},
    'Job Type': {0: 'unemployed / unskilled non-resident', 1: 'unskilled resident', 2: 'skilled employee',
                 3: 'management / self-employed / highly qualified'},
    'Has Telephone': {0: 'no', 1: 'yes'},
    'Foreign Worker': {0: 'yes', 1: 'no'},
}

def describe_value(feature, value):
    """Turn an encoded value back into a readable label."""
    return CODEBOOK.get(feature, {}).get(int(value), str(value))

X = df_encoded.drop('target', axis=1)
X.columns = feature_names
y = df_encoded['target']

print("✓ Features encoded and named")

# ── CELL 5: Train Random Forest Model ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("✓ Model trained")
print(f"\nModel Performance:")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Good Risk', 'Bad Risk'])}")

# ── CELL 6: SHAP Explainer Setup ─────────────────────────────
print("Setting up SHAP explainer...")

explainer = shap.TreeExplainer(model)

def bad_risk_shap(shap_output):
    """
    Return SHAP values for the 'Bad Risk' class (index 1) as (n_samples, n_features).
    Older SHAP returns a list [class0, class1]; SHAP ≥ 0.45 returns one
    array of shape (n_samples, n_features, n_classes).
    """
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1])
    shap_output = np.asarray(shap_output)
    if shap_output.ndim == 3:
        return shap_output[:, :, 1]
    return shap_output

shap_bad = bad_risk_shap(explainer.shap_values(X_test))

print("✓ SHAP explainer ready")

# ── CELL 7: Global Feature Importance ────────────────────────
shap.summary_plot(
    shap_bad,
    X_test,
    plot_type="bar",
    show=False,
    color='#2a78d6',
    plot_size=(12, 6)
)
plt.title('Global Feature Importance — What Drives Credit Risk Decisions',
          fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('global_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: global_importance.png")

# ── CELL 8: FAIRNESS ANALYSIS ─────────────────────────────────
# Descriptive check: average predicted risk per group on the test set.
# These are raw group averages — they do NOT control for other factors,
# so they flag where to look, not proof of discrimination.
print("\n" + "="*60)
print("FAIRNESS ANALYSIS — Average predicted risk by group")
print("="*60)

X_test_copy = X_test.copy()
X_test_copy['predicted_risk'] = y_prob
X_test_copy['Age Group'] = pd.cut(X_test_copy['Age'], bins=[0, 25, 35, 50, 120],
                                  labels=['≤ 25', '26–35', '36–50', '> 50'])

def print_group_risk(column, decode=True):
    print(f"\n{column}:")
    grouped = X_test_copy.groupby(column, observed=True)['predicted_risk'].agg(['mean', 'count'])
    for group, row in grouped.iterrows():
        label = describe_value(column, group) if decode else str(group)
        bar = "█" * int(row['mean'] * 20)
        print(f"  {label:38} | {bar:20} | {row['mean']:.3f}  (n={int(row['count'])})")

print_group_risk('Employment Duration')
print_group_risk('Age Group', decode=False)
print_group_risk('Personal Status & Sex')
print_group_risk('Foreign Worker')

print("""
NOTE:
Employment duration is a conventional credit risk factor, not a protected
characteristic. Age, sex and nationality-related attributes are the ones
relevant to EU non-discrimination law. Small groups (low n) give unstable
averages — treat differences as leads for further analysis.
""")

# ── CELL 9: Individual Applicant Explainer ────────────────────
def explain_applicant(applicant_data, applicant_name="Applicant"):
    """
    Takes a list of 20 encoded feature values and returns:
    - Risk score and decision
    - All factors ranked by SHAP impact (top 3 used in the report)
    """
    applicant_df = pd.DataFrame([applicant_data], columns=feature_names)

    risk_score = model.predict_proba(applicant_df)[0][1]
    decision = "DECLINED" if risk_score > 0.5 else "APPROVED"

    shap_individual = bad_risk_shap(explainer.shap_values(applicant_df))[0]

    factor_impact = list(zip(feature_names, shap_individual, applicant_data))
    factor_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        'name': applicant_name,
        'inputs': list(applicant_data),
        'risk_score': risk_score,
        'decision': decision,
        'all_factors': factor_impact,
        'top_factors': factor_impact[:3],
        'shap_values': shap_individual
    }

def audit_id(result):
    """Deterministic ID (Python's hash() changes between sessions)."""
    key = f"{result['name']}|{result['inputs']}"
    return "LNS-" + hashlib.sha256(key.encode()).hexdigest()[:8].upper()

def plain_language_report(result):
    """
    Template-based plain-language explanation (no LLM call).
    See Cell 12 for the LLM version.
    """
    decision = result['decision']
    score = result['risk_score']
    factors = result['top_factors']

    lines = []
    for i, (fname, shap_val, fval) in enumerate(factors, start=1):
        effect = "raised" if shap_val > 0 else "lowered"
        lines.append(f"  {i}. {fname}: {describe_value(fname, fval)}\n"
                     f"     → {effect} the estimated risk ({shap_val:+.3f})")
    rationale = "\n".join(lines)

    summary = ("Your application has been approved. The estimated probability of "
               "default is below the 50% decision threshold for your credit profile."
               if decision == "APPROVED" else
               "Your application has been declined. The assessment identified factors "
               "that indicate an elevated probability of default. The three most "
               "significant factors are listed above.")

    return f"""
╔══════════════════════════════════════════════════════════════╗
║           LOANSENSE — CREDIT DECISION REPORT                 ║
║           Prototype explanation (not a compliance cert.)     ║
╚══════════════════════════════════════════════════════════════╝

Applicant:    {result['name']}
Decision:     {decision}
Risk Score:   {score:.1%} estimated probability of default

DECISION RATIONALE
──────────────────
This decision was primarily influenced by three factors:

{rationale}

PLAIN LANGUAGE SUMMARY
───────────────────────
{summary}

YOUR RIGHTS
────────────
You may request an explanation of this decision and ask for it to be
reviewed by a person.

Audit ID: {audit_id(result)}
"""

# ── CELL 10: Run a Sample Applicant ──────────────────────────
# Encoded values — see CODEBOOK in Cell 4 for what each code means
sample_applicant = [
    1,      # Checking Account Status: 0–200 DM
    24,     # Loan Duration (months)
    2,      # Credit History: existing credits paid duly
    3,      # Loan Purpose: furniture
    5000,   # Loan Amount (DM)
    1,      # Savings Account: 100–500 DM
    2,      # Employment Duration: 1–4 years
    3,      # Installment Rate (1–4, share of disposable income)
    1,      # Personal Status & Sex: female, divorced/separated/married
    0,      # Other Debtors: none
    2,      # Residence Duration (1–4)
    2,      # Property Type: car or other
    35,     # Age
    2,      # Other Installment Plans: none
    1,      # Housing Type: own
    1,      # Existing Credits (count)
    2,      # Job Type: skilled employee
    1,      # Number of Dependents
    1,      # Has Telephone: yes
    0,      # Foreign Worker: yes
]

result = explain_applicant(sample_applicant, "Sample Applicant A")
print(plain_language_report(result))

# ── CELL 11: SHAP Bar Chart for Applicant ────────────────────
top8 = result['all_factors'][:8][::-1]   # reversed so the biggest sits on top
factor_names = [f"{f[0]} = {describe_value(f[0], f[2])}" for f in top8]
shap_vals = [f[1] for f in top8]
colors = ['#e34948' if v > 0 else '#2a78d6' for v in shap_vals]

plt.figure(figsize=(12, 6))
plt.barh(factor_names, shap_vals, color=colors, height=0.5)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.title(f'SHAP Explanation — {result["name"]}\nDecision: {result["decision"]} | Risk Score: {result["risk_score"]:.1%}',
          fontsize=12, pad=12)
plt.xlabel('SHAP value (red = increases risk, blue = decreases risk)')
plt.tight_layout()
plt.savefig('individual_explanation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: individual_explanation.png")

# ── CELL 12: LangChain Integration (Production Version) ──────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRODUCTION EXTENSION — LangChain + LLM Integration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In production, replace plain_language_report() with:

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["decision", "score", "factor1",
                         "factor2", "factor3"],
        template='''
        You are a bank compliance officer writing a credit
        decision explanation for a customer.

        Decision: {decision}
        Risk Score: {score}
        Top factors: {factor1}, {factor2}, {factor3}

        Write a clear, empathetic explanation in 3 sentences
        that a customer with no financial background can
        understand. Do not use technical jargon. Only mention
        the factors listed. Mention the right to human review.
        '''
    )

    chain = prompt | llm
    explanation = chain.invoke({
        "decision": result["decision"],
        "score": f"{result['risk_score']:.1%}",
        "factor1": factors[0][0],
        "factor2": factors[1][0],
        "factor3": factors[2][0],
    })
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("\n✓ LoanSense pipeline complete")
print("  Outputs: global_importance.png, individual_explanation.png")
