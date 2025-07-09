
# Customer Lifetime Value Prediction w/ BeteGeoFiiter

# pip install lifetimes
# pip install sqlalchemy
# pip install mysql-connector-python-rf
# pip install mysql
# pip install pymysql

import mysql
import pymysql
import mysql.connector
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Read .csv file:
# Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.shape

# Data Preparation:
df.describe().T
df.dropna(inplace=True)
df.isnull().sum()
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df.head()

# Accessing and Suppressing Outliers:

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

# BGNBD MODEL PREPARATION:
df = df[df["Country"] == "United Kingdom"]
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: ((date.max() - date.min()).days) / 7,
                                                         lambda date: ((today_date - date.min()).days) / 7],
                                         "Invoice": lambda freq: freq.nunique(),
                                         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary_value"]

cltv_df["monetary_value"] = cltv_df["monetary_value"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["monetary_value"] > 0]
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

# BGNBD MODEL:
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])

plot_period_transactions(bgf)
plt.show()

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])

# GAMMA GAMMA MODEL:
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_value"])

cltv_df["exp_avg_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                    cltv_df["monetary_value"])

# CLTV Prediction:
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary_value"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# Scaling and Segmentation:
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final["Segment"] = pd.qcut(cltv_final["scaled_clv"], 4, ["D", "C", "B", "A"])
cltv_final.to_csv("final_cltv_segmented.csv", index=False)
