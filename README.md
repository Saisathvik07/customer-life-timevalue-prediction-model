# customer-life-timevalue-prediction-model 
# 🧮 Customer Lifetime Value (CLTV) Prediction

This project predicts the **Customer Lifetime Value (CLTV)** using historical transaction data from the [Online Retail Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset). It applies **RFM analysis**, **BG/NBD model**, and **Gamma-Gamma model** to estimate the future value of customers over time.

---

## 📌 Problem Statement

Predict the future value of a customer to a business over the entire duration of their relationship, incorporating:
- Past purchase history
- Purchase frequency
- Monetary value of purchases

---

## 🗂 Dataset

- Source: Kaggle  
- Link: [Online Retail Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)  
- Format: Excel file (`Online Retail.xlsx`)  
- Contains transactions from a UK-based online retailer from 2010 to 2011.

---

## ⚙️ Features Used

- **Recency**: Days since last purchase
- **Frequency**: Total number of purchases
- **Monetary**: Total spending
- **Predicted Purchases**: Future transactions predicted using BG/NBD
- **Expected Monetary Value**: Avg. value per transaction using Gamma-Gamma
- **CLTV**: Predicted customer lifetime value for 3 months

---

## 🧠 Models Used

- 📉 **BetaGeoFitter (BG/NBD)** – Predicts future purchase frequency
- 💰 **GammaGammaFitter** – Predicts average purchase value

---

## 📁 Files Included

| File Name                | Description                                   |
|-------------------------|-----------------------------------------------|
| `cltv_prediction.py`     | Main Python script for data analysis & modeling |
| `cltv_predictions.csv`   | Final output with CLTV values per customer     |
| `README.md`              | Project documentation                         |

---
📌 Use Cases
Identify high-value customers

Personalize marketing campaigns

Improve customer retention strategies
