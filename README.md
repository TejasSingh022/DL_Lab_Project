# ICD-9 Code Prediction

A minimal Streamlit application using the existing **PLM-ICD MIMIC-III Full** trained model. It returns the ten highest sigmoid scores with ICD-9 codes and available descriptions. There is no training or fine-tuning.

## Current state

The application and `.venv` are created in this directory (`DL_Final_Project` is the project root; no extra nested `icd-prediction` folder is needed).

## Files

```text
DL_Final_Project/
├── PLM-ICD/                      # existing, unchanged repository
├── models/
│   └── roberta-mimic3-full/
│       ├── pytorch_model.bin     # trained Full model, including LAAT head
│       ├── config.json          # only one config.json is needed
│       ├── vocab.json           # matching pretrained tokenizer
│       └── merges.txt
├── data/
│   ├── ALL_CODES.txt
│   └── icd9_descriptions.json
├── src/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── preprocess.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Windows PowerShell setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, you can run without changing execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address Streamlit prints (normally `http://localhost:8501`), paste a discharge summary, and click **Predict**. The first successful request loads and caches the model. Press Ctrl+C in PowerShell to stop the app.
