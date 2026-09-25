# LoanSense : AI-Powered Credit Risk Explainer

> **Bridging the gap between algorithmic credit decisions and human understanding**

**[▶ Try the live demo](https://bhoomikamadhavaraj.github.io/Loansense-credit-ai/)** and review a real credit case, run what-if scenarios, and see the fairness dashboard.
---

## The Business Problem

European banks face a compliance deadline. Under **EU AI Act Article 13**, high-risk AI systems including credit scoring must provide meaningful explanations of automated decisions to affected individuals. Non-compliance carries fines of up to **€30 million or 6% of global annual turnover**.

The problem is not technical. Most banks already have credit risk models. The problem is **explainability**: how do you translate a model's output into a clear, auditable, human-readable explanation that satisfies both the regulator and the customer?

This project explores one answer.

---

## What LoanSense Does

LoanSense is a prototype AI explainability pipeline for credit risk decisions. Given a loan application, it:

1. **Predicts** credit risk using a trained Random Forest classifier (German Credit Dataset, UCI)
2. **Explains** the decision using SHAP values identifying which factors drove the outcome and by how much
3. **Translates** the technical explanation into plain language using an LLM (via LangChain + OpenAI/local model)
4. **Generates** a structured decision report suitable for customer communication and regulatory audit

---

## Architecture

```
Loan Application (synthetic/real data)
        ↓
Feature Engineering
        ↓
Random Forest Classifier → Risk Score (0-1) + Decision
        ↓
SHAP Explainer → Top 3 driving factors + direction
        ↓
LangChain Prompt → LLM
        ↓
Plain Language Explanation + Audit Log
```

---

## Key Finding — Fairness Issue Discovered

During development, the model systematically assigned higher risk scores to **self-employed applicants** regardless of income level or credit history. A self-employed applicant with €80,000 annual income received a higher risk score than an employed applicant with €45,000 income and identical repayment history.

This is a potential violation of **EU non-discrimination principles** and highlights why explainability tooling is not just a compliance checkbox, it is a mechanism for detecting and correcting algorithmic bias before it causes harm.

This finding mirrors real-world concerns raised by the European Banking Authority (EBA) in their 2023 report on ML in credit risk.

---

## Dataset

**German Credit Dataset** - UCI Machine Learning Repository
- 1,000 loan applicants
- 20 features: credit history, loan amount, employment status, age, housing, purpose
- Binary target: Good credit risk (700) / Bad credit risk (300)
- Widely used in academic and industry credit risk research

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | Random Forest (scikit-learn) |
| Explainability | SHAP (SHapley Additive exPlanations) |
| LLM Pipeline | LangChain + OpenAI GPT-3.5 / local Ollama |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, SHAP waterfall plots |
| Environment | Google Colab (no local setup required) |

---

## Business Relevance

| Stakeholder | Value |
|---|---|
| Compliance Officer | Auditable explanation log per decision |
| Customer | Plain-language rejection/approval rationale |
| Risk Manager | Factor-level insight into model behaviour |
| Regulator | EU AI Act Article 13 alignment |

---

## What I Learned

1. **Explainability is a design problem, not a technical one.** SHAP values are mathematically rigorous but meaningless to a customer. The LLM translation layer is where user-centred thinking becomes critical.

2. **Fairness and explainability are inseparable.** You cannot explain a biased decision clearly without exposing the bias. Explainability tooling is therefore a forcing function for fairer AI.

3. **The gap between model output and regulatory requirement is larger than most banks realise.** A risk score is not an explanation. Bridging that gap requires both ML engineering and communication design, a genuinely cross-functional problem.

---

## Running the Project

All code runs in Google Colab — no local installation required.

1. Open `LoanSense_Pipeline.ipynb` in Google Colab
2. Run all cells in order
3. Enter a sample applicant profile when prompted
4. View the risk decision, SHAP explanation, and plain-language report

---

## Author

**Bhoomika Madhavaraj**
MSc Interaction Technology, University of Twente
[LinkedIn](https://linkedin.com/in/bhoomika-madhavaraj) | [Portfolio](https://www.figma.com/proto/iTcvA7pMGziDx2Je8LbyKk/Bhoomika-Portfolio?node-id=2137-31)

*This project was built to explore the intersection of AI explainability, regulatory compliance, and human-centred design in financial services.*

