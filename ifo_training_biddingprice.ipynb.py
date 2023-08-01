# Databricks notebook source
# MAGIC %md
# MAGIC Train bidding prices for live test on 08/13/2023, need GPU cluster. 

# COMMAND ----------

# DBTITLE 1,Steps
#1. Install modules
#2. Load best float model
#3. Load previous year features
#4. Substitute real-time features with weekly features
#5. Derive campaign similarities
#6. Load selected audience 
#7. Inference for audience with weekly features
#8. Save biddingprice for audience

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
import numpy as np
import pandas as pd
import sklearn
import random
import time
import autogluon.core as ag
import matplotlib.pyplot as plt

from datetime import datetime
import pyspark.sql.functions as f
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

sc.setLogLevel("WARN")
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

# DBTITLE 1,Config model save & parameters
save_path_reg2 = "/dbfs/FileStore/shared_uploads/c146802@8451.com/reg_price_" + timestr + "/" # directories for biddingprice model 
print(timestr)

time_limit = 12*60*60  # train various models for ~12 hrs
num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
search_strategy = 'auto'  # to automatically tune hyperparameters using random search routine with a local scheduler

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

# DBTITLE 1,Train biddingprice model
sampled_opt_1 = spark.read.parquet("abfss://users@sa8451labsdev.dfs.core.windows.net/c146802/inflight/sampled_opt_1_061420232025.parquet")
sampled = sampled_opt_1
col = list(sampled.columns)
del_col = ["CAMP_END_DATE", "ehhn", "ExpDT", "TDID", "RenderingContext", "KPMID", "ImpressionId", "CAMP_START_DATE", "CampaignId", "CampaignName", "AdvertiserId", "LogEntryTime", "id", "CUST_IDX", "dense_rank", "MatchedFoldPosition", "emb", "TTDCostInUSDollars", 'PartnerCostInUSDollars', "IDLS", "ED", "AdvertiserCostInUSDollars", "KPM_PROJECT_ID", "cycle_date", "TTD_COOKIES", "effo_hh_id", "Recency",  "Device_ID", 'DataUsageTotalCost', "trn_dt", "trn_tm", "gtin_no", "dense_rank", "HSHD_CODE", "campaign_id", "UPC_ID", "ngr_campainid", "sales_conversion_ori", "FeeFeaturesCost", 'clickthrough_conversion', 'viewthrough_nonclk_conversion', 'sales_conversion', 'net_spend_amt', "cust_visits"]

header = [c for c in col if c not in del_col]
s5_test = sampled.select(header).toPandas()

s5_features = ["DeviceMake", "Region", "DeviceModel", "DeviceType", "Browser", "SupplyVendor", "AuctionType", "AdFormat", "Site"]
s5=s5_test
s5.rename(columns={"MediaCostInBucks": "biddingprice"}, inplace=True) #np.random.uniform(0.002, 0.004, [s3.shape[0],1])
#s5.rename(columns={"clickthrough_conversion": "ctr_conversion"}, inplace=True)
#s5.rename(columns={"viewthrough_nonclk_conversion": "vt_conversion"}, inplace=True)
#s5.rename(columns={"sales_conversion": "sales_conversion"}, inplace=True)
#s5.rename(columns={"net_spend_amt": "spend_amount"}, inplace=True)

s5_encoder = OrdinalEncoder(cols=s5_features, handle_unknown="impute").fit(s5)
s5 = s5_encoder.transform(s5)

sz_s5 = s5.shape[0]
for i, h in enumerate(header):
    if h == "MediaCostInBucks":
        header[i] = "biddingprice"
        
df = pd.DataFrame(np.random.randint(1, 5, size=(sz_s5, 1)), columns=["AuctionType"])
for h in header:
    df[h] = np.random.randint(1, 7, size=(sz_s5, 1))

df["biddingprice"] = np.random.uniform(2, 4, [sz_s5,1])

encoder = OrdinalEncoder(cols=s5_features, handle_unknown="impute").fit(df)
df = encoder.transform(df)

for h in header:
    df[h] = s5[h]

df["biddingprice"]=s5["biddingprice"].astype("float") - 0.0006 + random.uniform(0, 0.0008) # act like SecondPrice Auction

# COMMAND ----------

df = df.sample(frac=1).reset_index(drop=True) #shuffle rows
sz = df.shape[0]
train_data = df.iloc[0:int(sz*0.60),:]
val_data = df.iloc[int(sz*0.60):int(sz*0.80),:]
test_data = df.iloc[int(sz*0.80):,:]

y_test_reg = test_data["biddingprice"]
test_data_nolabel_reg = test_data.drop(columns=["biddingprice"])

# COMMAND ----------

nn_options = { 
    'num_epochs': 500,  
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),  
}

hyperparameters = { 
#                   'LinearModel': lm_options,
                   'RF': {"n_estimators": 100},
                   'XT': {"n_estimators": 300},
                   'CAT': {"iterations": 10000},
                   'GBM': {"num_boost_round": 10000},
                   'NN': nn_options, 
                  }  

predictor_reg_r2 = TabularPredictor(label="biddingprice", eval_metric=my_scorer_reg_r2, path=save_path_reg2).fit(
    train_data, tuning_data=val_data, time_limit=time_limit, ag_args_fit ={"num_gpus":1},
    hyperparameters=hyperparameters, 
)

# COMMAND ----------

# DBTITLE 1,Load biddingprice model
save_path_reg_r2_latest = "/dbfs/FileStore/shared_uploads/c146802@8451.com/reg2_0730202314"
predictor_reg_r2_latest = TabularPredictor.load(save_path_reg_r2_latest)
predictor_information = predictor_reg_r2_latest.info()
print(predictor_information["features"])
