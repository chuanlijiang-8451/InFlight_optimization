# Databricks notebook source
# MAGIC %md
# MAGIC Predict sales conversion for live test on 08/13/2023, need GPU cluster. 

# COMMAND ----------

# DBTITLE 1,Steps
# Install/import modules
# Load sales conversion model
# Load live-test selected audience
# Load imputted 
# Inference audience sales conversion probabilities
# Save to csv file

# COMMAND ----------

# DBTITLE 1,Install modules
!pip install -U pip
!pip install -U setuptools wheel
#!pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html #cpu
!pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 #gpu
!pip install  autogluon==0.5.0 autogluon.tabular "mxnet<2.0.0"

!/databricks/python3/bin/python -m pip install --no-cache-dir --use-deprecated=legacy-resolver --upgrade pip
!pip install --no-cache-dir --use-deprecated=legacy-resolver --upgrade category_encoders
!pip install pyspark_dist_explore

# COMMAND ----------

# DBTITLE 1,Import modules: GPU cluster
import os
import re
import numpy as np
import pandas as pd
import sklearn
import random
import time
import autogluon.core as ag
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta
from pyspark.sql.window import Window
from pyspark_dist_explore import hist
from autogluon.core.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, precision_score
from autogluon.tabular import TabularDataset, TabularPredictor
from category_encoders import OrdinalEncoder
from pyspark.sql.functions import col, floor, lit, concat, broadcast, countDistinct, sum, avg, max, count, concat_ws, array, rand
from pyspark.sql.functions import substring, regexp_replace, when, date_add
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, log_loss
from sklearn.metrics import cohen_kappa_score, precision_recall_curve, auc

pd.options.mode.chained_assignment = None  # default='warn'
timestr = time.strftime("%m%d%Y%H")
print(timestr)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# COMMAND ----------

# DBTITLE 1,Import modules: CPU cluster
!pip install --no-cache-dir --use-deprecated=legacy-resolver --upgrade category_encoders
import re
import time
import random
import argparse
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as f

from datetime import date
from pyspark.sql import Window
from IPython.display import Image
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from effodata import ACDS, golden_rules
from pyspark.sql.functions import col, floor, lit, concat, broadcast, countDistinct
from pyspark.sql.functions import col, floor, lit, concat, broadcast, countDistinct, sum, avg, max, count, concat_ws, array, rand
timestr = time.strftime("%m%d%Y%H")
print(timestr)

# COMMAND ----------

# DBTITLE 1,Config spark for labs
#kv-8451-mbx-comislty-prd in GBX1-dev
dbutils.secrets.list("kv-8451-labs-dev")

appID = dbutils.secrets.get(scope = "kv-8451-labs-dev", key = "spLabs-app-id")
password = dbutils.secrets.get(scope = "kv-8451-labs-dev", key = "spLabs-pw")
tenantID = "5f9dc6bd-f38a-454a-864c-c803691193c5"

spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id", appID)
spark.conf.set("fs.azure.account.oauth2.client.secret", password)
spark.conf.set("fs.azure.account.oauth2.client.endpoint", "https://login.microsoftonline.com/" + tenantID + "/oauth2/token")

# COMMAND ----------

# DBTITLE 1,Customize metrics: regressor
def customize_reg_score_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2
 
def customize_reg_score_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

my_scorer_reg_r2 = make_scorer("customize_reg_score_r2", score_func = customize_reg_score_r2, greater_is_better=True)
my_scorer_reg_mse = make_scorer("customize_reg_score_mse", score_func = customize_reg_score_mse, greater_is_better=True)

# COMMAND ----------

# DBTITLE 1,Customize metrics: cls
def customize_cls_score_bacc(y_true, y_pred):
    bacc = balanced_accuracy_score(y_true, y_pred)
    return bacc

def customize_cls_score_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc
  
def customize_cls_score_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1

def customize_cls_score_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision
  
def customize_cls_score_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall
  
def customize_cls_score_avg_precision(y_true, y_pred):
    avg_precision = average_precision_score(y_true, y_pred)
    return avg_precision
  
my_scorer_cls_bacc = make_scorer("customize_cls_score_bacc", score_func = customize_cls_score_bacc, greater_is_better=True)
my_scorer_cls_auc = make_scorer("customize_cls_score_auc", score_func = customize_cls_score_auc, greater_is_better=True)
my_scorer_cls_f1 = make_scorer("customize_cls_score_f1", score_func = customize_cls_score_f1, greater_is_better=True)
my_scorer_cls_precision = make_scorer("customize_cls_score_precision", score_func = customize_cls_score_precision, greater_is_better=True)
my_scorer_cls_recall = make_scorer("customize_cls_score_recall", score_func = customize_cls_score_recall, greater_is_better=True)
my_scorer_cls_avg_precision = make_scorer("customize_cls_score_avg_precision", score_func = customize_cls_score_avg_precision, greater_is_better=True)

# COMMAND ----------

# DBTITLE 1,Load sales conversion model
save_path_cls_latest = "/dbfs/FileStore/shared_uploads/c146802@8451.com/cls_0803202307"
predictor_cls_ap_latest = TabularPredictor.load(save_path_cls_latest)
predictor_information = predictor_cls_ap_latest.info()
print(predictor_information["features"])

# COMMAND ----------

# DBTITLE 1,Load inference vectors: after imputing
inference_vector_IFO4_pd = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_IFO4_0803202399.csv")

inference_vector_IFO5_pd = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_IFO5_0803202399.csv")

inference_vector_IFO6_pd = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_IFO6_0803202399.csv")

# COMMAND ----------

# DBTITLE 1,Encode inference vectors
inf_features = ["DeviceMake", "Region", "DeviceModel", "DeviceType", "Browser", "SupplyVendor", "AuctionType", "AdFormat", "Site"]

inf_encoder_IFO4 = OrdinalEncoder(cols=inf_features, handle_unknown="impute").fit(inference_vector_IFO4_pd)
inf_vec_IFO4 = inf_encoder_IFO4.transform(inference_vector_IFO4_pd)
#inf_vec_IFO4.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_encoder_IFO4_0803202399.csv")

inf_encoder_IFO5 = OrdinalEncoder(cols=inf_features, handle_unknown="impute").fit(inference_vector_IFO5_pd)
inf_vec_IFO5 = inf_encoder_IFO5.transform(inference_vector_IFO5_pd)
#inf_vec_IFO5.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_encoder_IFO5_0803202399.csv")

inf_encoder_IFO6 = OrdinalEncoder(cols=inf_features, handle_unknown="impute").fit(inference_vector_IFO6_pd)
inf_vec_IFO6 = inf_encoder_IFO6.transform(inference_vector_IFO6_pd)
#inf_vec_IFO6.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_encoder_IFO6_0803202399.csv")

# COMMAND ----------

# DBTITLE 1,Load encoded inference vectors
inf_vec_IFO4 = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_encoder_IFO4_0803202399.csv")

inf_vec_IFO5 = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_encoder_IFO5_0803202399.csv")

inf_vec_IFO6 = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inference_vector_encoder_IFO6_0803202399.csv")

# COMMAND ----------

# DBTITLE 1,Inference
j = 1 # week 

#IFO4
sc_prob = predictor_cls_ap_latest.predict_proba(inf_vec_IFO4, transform_features = True)[1]
inf_sc_prob = pd.DataFrame(inf_vec_IFO4["ehhn"].values, columns=["ehhn"])
inf_sc_prob["sales_conversion"] = sc_prob.values 
inf_sc_prob.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/IFO4_inference_salesconversion_prob_week_1_0803202399.csv", index=False)

#IFO5
sc_prob = predictor_cls_ap_latest.predict_proba(inf_vec_IFO5, transform_features = True)[1]
inf_sc_prob = pd.DataFrame(inf_vec_IFO5["ehhn"].values, columns=["ehhn"])
inf_sc_prob["sales_conversion"] = sc_prob.values 
inf_sc_prob.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/IFO5_inference_salesconversion_prob_week_1_0803202399.csv", index=False)

#IFO6
sc_prob = predictor_cls_ap_latest.predict_proba(inf_vec_IFO6, transform_features = True)[1]
inf_sc_prob = pd.DataFrame(inf_vec_IFO6["ehhn"].values, columns=["ehhn"])
inf_sc_prob["sales_conversion"] = sc_prob.values 
inf_sc_prob.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/IFO6_inference_salesconversion_prob_week_1_0803202399.csv", index=False)

# COMMAND ----------

# DBTITLE 1,Load sales conversion probabilities 
inf_sc_prob_IFO4 = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/IFO4_inference_salesconversion_prob_week_1_0803202399.csv")

inf_sc_prob_IFO5 = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/IFO5_inference_salesconversion_prob_week_1_0803202399.csv")

inf_sc_prob_IFO6 = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/IFO6_inference_salesconversion_prob_week_1_0803202399.csv")

# COMMAND ----------

# DBTITLE 1,Load ngr files
ngr_0623 = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/ngr_0623_0801202308.parquet").toPandas()

ngr_IFO4 = ngr_0623.merge(inf_sc_prob_IFO4, on="ehhn")
print(ngr_IFO4[ngr_IFO4["sales_conversion"]>0.5].shape[0]/ngr_IFO4.shape[0])

ngr_IFO5 = ngr_0623.merge(inf_sc_prob_IFO5, on="ehhn")
print(ngr_IFO5[ngr_IFO5["sales_conversion"]>0.5].shape[0]/ngr_IFO5.shape[0])

ngr_IFO6 = ngr_0623.merge(inf_sc_prob_IFO6, on="ehhn")
print(ngr_IFO6[ngr_IFO6["sales_conversion"]>0.5].shape[0]/ngr_IFO6.shape[0])

# COMMAND ----------

# DBTITLE 1,Plot scatterplot sales conversion probabilities vs ngr_score
plt.figure(figsize=(10,10))
p4 = plt.scatter(ngr_IFO4["sales_conversion"], ngr_IFO4["ngr_score"], color='blue')
p5 = plt.scatter(ngr_IFO5["sales_conversion"], ngr_IFO5["ngr_score"], color='orange')
p6 = plt.scatter(ngr_IFO6["sales_conversion"], ngr_IFO6["ngr_score"], color='green')
plt.legend((p4, p5, p6), ("IFO4", "IFO5", "IFO6"))
plt.xlabel("Sales conversion probability")
plt.ylabel("ngr_score")
plt.savefig("/dbfs/FileStore/shared_uploads/c146802@8451.com/sales_conversion_prob_08043202399.png")

# COMMAND ----------

#https://adb-291758323461480.0.azuredatabricks.net/files/shared_uploads/c146802@8451.com/sales_conversion_prob_08043202399.png/?o=291758323461480
