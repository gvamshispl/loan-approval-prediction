from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Train model at startup
data = pd.read_csv("loan.csv")

X = data[
    ["Age","AnnualIncome","CreditScore","LoanAmount",
     "CreditCardUtilizationRate","SavingsAccountBalance",
     "CheckingAccountBalance","TotalAssets","MonthlyIncome",
     "NetWorth","InterestRate","MonthlyLoanPayment",
     "TotalDebtToIncomeRatio","RiskScore"]
]
y = data["LoanApproved"]

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    input_data = {
        "Age": float(form_data["Age"]),
        "AnnualIncome": float(form_data["AnnualIncome"]),
        "CreditScore": float(form_data["CreditScore"]),
        "LoanAmount": float(form_data["LoanAmount"]),
        "CreditCardUtilizationRate": float(form_data["CreditCardUtilizationRate"]),
        "SavingsAccountBalance": float(form_data["SavingsAccountBalance"]),
        "CheckingAccountBalance": float(form_data["CheckingAccountBalance"]),
        "TotalAssets": float(form_data["TotalAssets"]),
        "MonthlyIncome": float(form_data["MonthlyIncome"]),
        "NetWorth": float(form_data["NetWorth"]),
        "InterestRate": float(form_data["InterestRate"]),
        "MonthlyLoanPayment": float(form_data["MonthlyLoanPayment"]),
        "TotalDebtToIncomeRatio": float(form_data["TotalDebtToIncomeRatio"]),
        "RiskScore": float(form_data["RiskScore"])
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    result = "Loan Approved ✅" if prediction == 1 else "Loan Not Approved ❌"

    return render_template(
        "index.html",
        result=result,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
