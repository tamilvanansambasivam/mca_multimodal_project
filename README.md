# **Project Setup Guide**

## **1. Clone the Repository**

```bash
git clone https://github.com/tamilvanansambasivam/mca_multimodal_project.git
cd mca_multimodal_project
```

## **2. Create & Activate Virtual Environment**

### Linux / macOS / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## **3. Upgrade Packaging Tools**

```bash
python -m pip install --upgrade pip setuptools wheel
```

## **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

## **5. Verify Installed Libraries**

```bash
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import pandas, numpy, sklearn, altair; print('Core libs OK')"
```

## **6. Generate Data & Train Models**

```bash
mkdir models
python data_gen.py
python train.py
```

## **7. Run the Application**

```bash
streamlit run app.py
```

---
