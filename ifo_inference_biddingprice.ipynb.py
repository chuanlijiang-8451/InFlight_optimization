# Databricks notebook source
# MAGIC %md
# MAGIC Predict bidding prices for live test on 08/13/2023, need GPU cluster. 

# COMMAND ----------

# DBTITLE 1,Steps
# Install/import modules
# Load best fluid model
# Load live-test selected audience
# Load features
# Substitute real-time features with weekly features
# Inference bidding price for audience
# Save biddingprice

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

# DBTITLE 1,Load best fluid bidding price model
save_path_reg_r2_latest = "/dbfs/FileStore/shared_uploads/c146802@8451.com/reg2_0730202306" 
predictor_reg2_r2_latest = TabularPredictor.load(save_path_reg_r2_latest)

# COMMAND ----------

# DBTITLE 1,Load inference vectors
inf_vec = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/inf_vec_0730202313.csv")

# COMMAND ----------

# DBTITLE 1,Inference
week = 1
bp = predictor_reg2_r2_latest.predict(inf_vec, transform_features = True)
inf_bp = pd.DataFrame(inf_vec["ehhn"].values, columns=["ehhn"])
inf_bp["basebid"] = bp.values
inf_bp.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/IFO_inference_biddingprice_week_" + str(week) + "_" + timestr + ".csv", index=False)
