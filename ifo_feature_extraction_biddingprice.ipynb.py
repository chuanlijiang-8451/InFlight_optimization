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

# DBTITLE 1,Load ngr, fsb: sometime ran into permission issues
ehhn_tdid_0623 = spark.read.parquet("abfss://ifo@sa8451labsdev.dfs.core.windows.net/inflight_campaigns/62640/ehhn_tdid/")
ngr_0623 = spark.read.parquet("abfss://ifo@sa8451labsdev.dfs.core.windows.net/inflight_campaigns/62640/ngr_scores/")
fsb_0623 = spark.read.parquet("abfss://ifo@sa8451labsdev.dfs.core.windows.net/inflight_campaigns/62640/fsb_engagement_scores/")
imp = spark.read.parquet("abfss://ifo@sa8451labsdev.dfs.core.windows.net/inflight_campaigns/62640/impressions/")

ehhn_tdid_0623 = ehhn_tdid_0623.withColumnRenamed("HSHD_CODE", "ehhn")
ngr_0623 = ngr_0623.withColumnRenamed("HSHD_CODE", "ehhn")
ngr_0623 = ngr_0623.withColumnRenamed("campaign_score", "ngr_score")
fsb_0623 = fsb_0623.withColumnRenamed("HSHD_CODE", "ehhn")
imp = imp.withColumnRenamed("HSHD_CODE", "ehhn")

ngr_0623.write.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/ngr_0623_" + timestr + ".parquet")
fsb_0623.write.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/fsb_0623_" + timestr + ".parquet")
imp.write.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/imp_" + timestr + ".parquet")
ehhn_tdid_0623.write.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/ehhn_tdid_0623_" + timestr + ".parquet")

# COMMAND ----------

# DBTITLE 1,Load above ngr, fsb
ngr_0623 = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/ngr_0623_0730202301.parquet")
fsb_0623 = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/fsb_0623_0730202301.parquet")
imp = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/imp_0730202301.parquet")
ehhn_tdid_0623 = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/ehhn_tdid_0623_0730202301.parquet")

# COMMAND ----------

# DBTITLE 1,Load REDs: 2022-2023
media_fd_2023 = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/media_fd_2023_072220230641.parquet")
media_fd_2022= spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/media_fd_2022.parquet")
reds_features = ['AuctionType',
 'ehhn',
 'LogEntryTime',
 'DeviceMake',
 'DeviceModel',
 'DeviceType',
 'Browser',
 'UserHourOfWeek',
 'SupplyVendor',
 'TemperatureInCelsius',
 'AdFormat',
 'Region',
 'Site']
reds_2223 = media_fd_2022.union(media_fd_2023).select(reds_features)

# COMMAND ----------

# DBTITLE 1,load: media engagement features: latest cycle_date: 06/25/2023
med_0623 = spark.read.parquet("abfss://media@sa8451camdev.dfs.core.windows.net/media_engagement_features/compiled/channel_type=DISPLAY/cycle_date=2023-06-25/")

# COMMAND ----------

# DBTITLE 1,Save audience file
reds_2223_target_ehhn = reds_2223.select("ehhn").intersect(ngr_0623.select("ehhn"))
reds_2223_target_ehhn_list = reds_2223_target_ehhn.select("ehhn").rdd.flatMap(lambda x: x).collect()
target_ehhn = pd.DataFrame(reds_2223_target_ehhn_list, columns=["ehhn"])
#target_ehhn.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/reds_2223_target_ehhn_list_9M_" + timestr + ".csv")
target_ehhn = pd.read_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/reds_2223_target_ehhn_list_9M_0729202321.csv")

# COMMAND ----------

# DBTITLE 1,Filter REDs records for target audience
reds_2223_target = reds_2223.join(f.broadcast(reds_2223_target_ehhn), reds_2223.columns[1:2])
reds_2223_target.write.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/reds_2223_target_" + timestr + ".parquet")
reds_2223_target_latest_tmp = reds_2223_target.withColumn("rn", f.row_number()
        .over(Window.partitionBy("ehhn")
        .orderBy(f.col("LogEntryTime").desc())))

reds_2223_target_latest = reds_2223_target_latest_tmp.filter(f.col("rn") == 1).drop("rn")
reds_2223_target_latest.write.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/reds_2223_target_latest_" + timestr + ".parquet")
print(timestr)

# COMMAND ----------

# DBTITLE 1,REDs, ngr, media, fsb
reds_2223_target_latest = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/reds_2223_target_latest_0730202304.parquet")
reds_ehhn = reds_2223_target_latest.select("ehhn").distinct()
ngr_ehhn = ngr_0623.select("ehhn").distinct()
med_ehhn = med_0623.select("ehhn").distinct()
fsb_ehhn = fsb_0623.select("ehhn").distinct()

reds_ngr = reds_ehhn.intersect(ngr_ehhn)
reds_med = reds_ehhn.intersect(med_ehhn)
reds_fsb = reds_ehhn.intersect(fsb_ehhn)

# COMMAND ----------

# DBTITLE 1,Join REDs, ngr, med: inference vectors
reds_ngr = reds_2223_target_latest.join(ngr_0623, reds_2223_target_latest.ehhn == ngr_0623.ehhn, "inner").drop(ngr_0623.ehhn)
reds_ngr_med = reds_ngr.join(med_0623, reds_ngr.ehhn == med_0623.ehhn, "inner").drop(med_0623.ehhn)
inf_vec_pd = reds_ngr_med.toPandas()
inf_vec_pd_dropna = inf_vec_pd.dropna()
inf_features = ["DeviceMake", "Region", "DeviceModel", "DeviceType", "Browser", "SupplyVendor", "AuctionType", "AdFormat", "Site"]
inf_encoder = OrdinalEncoder(cols=inf_features, handle_unknown="impute").fit(inf_vec_pd_dropna)
inf_vec = inf_encoder.transform(inf_vec_pd_dropna)
inf_vec.to_csv("/dbfs/FileStore/shared_uploads/c146802@8451.com/ifo_feature_extraction_biddingprice_" + timestr + ".csv")
