import os
import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

MODEL_PATH = "models/income_models.joblib"

CATEGORIES = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                  "Local-gov", "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Masters",
                  "9th", "Doctorate", "5th-6th", "10th", "7th-8th", "12th",
                  "1st-4th", "Prof-school", "Assoc-acdm", "Assoc-voc", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married",
                       "Separated", "Widowed", "Married-spouse-absent",
                       "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales",
                   "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                   "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                   "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family",
                     "Other-relative", "Unmarried"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico",
                       "Canada", "Germany", "India", "Japan", "Greece", "China",
                       "South", "Cuba", "Iran", "Honduras", "Philippines",
                       "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
                       "Ireland", "France", "Dominican-Republic", "Laos",
                       "Ecuador", "Taiwan", "Haiti", "Portugal", "Columbia",
                       "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
                       "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
                       "Guatemala", "Hungary", "Holand-Netherlands"]
}

NUMERIC_FIELDS = [
    ("age", "Age", "39"),
    ("fnlwgt", "Final Weight", "77516"),
    ("education-num", "Education Number", "13"),
    ("capital-gain", "Capital Gain", "2174"),
    ("capital-loss", "Capital Loss", "0"),
    ("hours-per-week", "Hours per Week", "40"),
]


class IncomeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Adult Income Prediction")
        self.root.geometry("760x720")
        self.root.resizable(False, False)

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model not found",
                "Run train_model.py first to create models/income_models.joblib."
            )
            root.destroy()
            return

        self.bundle = joblib.load(MODEL_PATH)
        self.entries = {}

        title = ttk.Label(root, text="Adult Income Prediction",
                          font=("Arial", 20, "bold"))
        title.pack(pady=12)

        subtitle = ttk.Label(
            root,
            text="Logistic Regression and KNN | Kaggle Adult Census Income"
        )
        subtitle.pack(pady=(0, 10))

        form = ttk.Frame(root)
        form.pack(fill="x", padx=35)

        row = 0
        for key, label, default in NUMERIC_FIELDS:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5)
            entry = ttk.Entry(form, width=35)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="w", padx=15, pady=5)
            self.entries[key] = entry
            row += 1

        for key in ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]:
            ttk.Label(form, text=key.replace("-", " ").title()).grid(
                row=row, column=0, sticky="w", pady=5
            )
            combo = ttk.Combobox(form, values=CATEGORIES[key],
                                 state="readonly", width=32)
            combo.current(0)
            combo.grid(row=row, column=1, sticky="w", padx=15, pady=5)
            self.entries[key] = combo
            row += 1

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=18)

        ttk.Button(button_frame, text="Predict Income",
                   command=self.predict).grid(row=0, column=0, padx=8)
        ttk.Button(button_frame, text="Load Sample",
                   command=self.load_sample).grid(row=0, column=1, padx=8)
        ttk.Button(button_frame, text="Clear",
                   command=self.clear).grid(row=0, column=2, padx=8)

        self.result = tk.StringVar(value="Prediction will appear here.")
        ttk.Label(root, textvariable=self.result,
                  font=("Arial", 14, "bold"),
                  wraplength=680,
                  justify="center").pack(pady=12)

        metrics = self.bundle.get("results", {})
        metric_text = []
        for name, r in metrics.items():
            metric_text.append(
                f"{name}: Accuracy={r['accuracy']:.2%}, "
                f"Precision={r['precision']:.2%}, Recall={r['recall']:.2%}"
            )
        ttk.Label(root, text="\n".join(metric_text),
                  justify="left").pack(pady=5)

    def load_sample(self):
        sample = {
            "age": "39",
            "fnlwgt": "77516",
            "education-num": "13",
            "capital-gain": "2174",
            "capital-loss": "0",
            "hours-per-week": "40",
            "workclass": "Private",
            "education": "Bachelors",
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "native-country": "United-States"
        }
        for key, value in sample.items():
            widget = self.entries[key]
            if isinstance(widget, ttk.Combobox):
                widget.set(value)
            else:
                widget.delete(0, tk.END)
                widget.insert(0, value)

    def clear(self):
        for key, widget in self.entries.items():
            if isinstance(widget, ttk.Combobox):
                widget.current(0)
            else:
                widget.delete(0, tk.END)
        self.result.set("Prediction will appear here.")

    def get_input(self):
        data = {}
        for key, widget in self.entries.items():
            value = widget.get().strip()
            if key in ["age", "fnlwgt", "education-num",
                       "capital-gain", "capital-loss", "hours-per-week"]:
                data[key] = float(value)
            else:
                data[key] = value
        return pd.DataFrame([data])

    def predict(self):
        try:
            sample = self.get_input()
            predictions = []

            for name, model in self.bundle["models"].items():
                pred = int(model.predict(sample)[0])
                label = ">50K" if pred == 1 else "<=50K"
                predictions.append(f"{name}: {label}")

            self.result.set("Prediction Result\n" + "\n".join(predictions))
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    IncomeGUI(root)
    root.mainloop()
