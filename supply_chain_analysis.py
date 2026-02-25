#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("C:/Users/adimi/Downloads/archive/supply_chain_disruption_recovery.csv")

print (df.head())
print (df.info())


# In[2]:


print (df.isnull().sum())


# In[3]:


from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://postgres:Hathi-Dada@localhost:5432/supply_chain_analytics"
)

df.to_sql(
    "supply_chain",
    engine,
    if_exists = "replace",
    index = False
)

print("Data loaded successfully!")


# ## Let's prepaer the Data

# In[3]:


X = df.drop([
    "disruption_id",
    "full_recovery_days",
    "partial_recovery_days",
    "revenue_loss_usd",
    "permanent_supplier_change"
], axis=1)

y = df["full_recovery_days"]

X = pd.get_dummies(X, drop_first=True)


# ## And train the model

# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))


# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = df.drop([
    "disruption_id",
    "response_type",
    "full_recovery_days",
    "partial_recovery_days"
], axis=1)

y = df["response_type"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


# ## PYTHON-BASED WHAT-IF SIMULATION (ADVANCED)

# In[6]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("C:/Users/adimi/Downloads/archive/supply_chain_disruption_recovery.csv")

X = df.drop([
    "disruption_id",
    "full_recovery_days",
    "partial_recovery_days",
    "revenue_loss_usd",
    "permanent_supplier_change"
], axis=1)

y = df["full_recovery_days"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# In[7]:


def simulate_scenario(
    data,
    severity_multiplier=1.0,
    response_time_reduction=0.0,
    add_backup=False
):
    scenario = data.copy()

    scenario["disruption_severity"] *= severity_multiplier
    scenario["response_time_days"] *= (1 - response_time_reduction)

    if add_backup:
        scenario["has_backup_supplier"] = True

    scenario = pd.get_dummies(scenario, drop_first=True)
    scenario = scenario.reindex(columns=X.columns, fill_value=0)

    predictions = model.predict(scenario)
    return predictions.mean()


# In[8]:


baseline = simulate_scenario(X)
high_severity = simulate_scenario(X, severity_multiplier=1.3)
backup_added = simulate_scenario(X, add_backup=True)
fast_response = simulate_scenario(X, response_time_reduction=0.3)

print("Baseline Recovery:", baseline)
print("High Severity:", high_severity)
print("Backup Supplier:", backup_added)
print("Faster Response:", fast_response)


# ## BUILD INDEX IN PYTHON (ML-Friendly)

# In[9]:


from sklearn.preprocessing import MinMaxScaler

cols = [
    "disruption_severity",
    "production_impact_pct",
    "full_recovery_days"
]

scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])

df["resilience_index"] = (
    (1 - df["disruption_severity"]) * 0.30 +
    (1 - df["production_impact_pct"]) * 0.25 +
    (1 - df["full_recovery_days"]) * 0.30 +
    df["has_backup_supplier"].astype(int) * 0.15
) * 100


# ## BONUS: PREDICT FUTURE RESILIENCE

# In[10]:


from sklearn.ensemble import RandomForestRegressor

X = df.drop([
    "resilience_index",
    "disruption_id"
], axis=1)

X = pd.get_dummies(X, drop_first=True)
y = df["resilience_index"]

model = RandomForestRegressor()
model.fit(X, y)


# In[16]:


get_ipython().system('pip install pandas numpy psycopg2-binary sqlalchemy scikit-learn joblib matplotlib')


# In[11]:


import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
engine = create_engine(
    "postgresql://postgres:Hathi-Dada@localhost:5432/supply_chain_analytics"
)

query = """
SELECT
    disruption_type,
    industry,
    supplier_tier,
    supplier_region,
    supplier_size,
    has_backup_supplier,
    disruption_severity,
    production_impact_pct,
    revenue_loss_usd,
    response_time_days,
    full_recovery_days
FROM supply_chain;
"""

df = pd.read_sql(query, engine)

print(df.head())


# ## Feature Engineering

# In[12]:


import numpy as np

# Binary encoding
df["has_backup_supplier"] = df["has_backup_supplier"].astype(int)

# Log transform (financial data skew)
df["log_revenue_loss"] = np.log1p(df["revenue_loss_usd"])

# Target
y = df["full_recovery_days"]

# Features
X = df.drop(columns=["full_recovery_days", "revenue_loss_usd"])


# ## Train-Test Split

# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ## ML Pipeline (Preprocessing + Model)

# In[14]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Column types
numeric_features = [
    "supplier_tier",
    "disruption_severity",
    "production_impact_pct",
    "response_time_days",
    "log_revenue_loss"
]

categorical_features = [
    "disruption_type",
    "industry",
    "supplier_region",
    "supplier_size"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ## Train the Model

# In[15]:


pipeline.fit(X_train, y_train)


# ## Model Evolution

# In[16]:


from sklearn.metrics import mean_absolute_error, r2_score

y_pred = pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# In[17]:


import numpy as np

feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = pipeline.named_steps["model"].feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_importance.head(10))


# In[18]:


import joblib

joblib.dump(pipeline, "supply_chain_recovery_model.pkl")


# In[19]:


new_event = pd.DataFrame([{
    "disruption_type": "Geopolitical",
    "industry": "Automotive",
    "supplier_tier": 3,
    "supplier_region": "Asia",
    "supplier_size": "Large",
    "has_backup_supplier": 0,
    "disruption_severity": 8,
    "production_impact_pct": 60,
    "response_time_days": 14,
    "log_revenue_loss": np.log1p(1200000)
}])

prediction = pipeline.predict(new_event)
print("Predicted Full Recovery Days:", round(prediction[0], 1))


# In[20]:


df_results = X_test.copy()
df_results["actual_recovery_days"] = y_test
df_results["predicted_recovery_days"] = y_pred

df_results.to_sql(
    "ml_predictions",
    engine,
    if_exists="replace",
    index=False
)


# In[21]:


import joblib

model = joblib.load(
    r"C:/Users/adimi/Downloads/Data Analytics/Supply Chain App/supply_chain_recovery_model.pkl"
)

print("Model loaded successfully!")


# In[22]:


print(type(model))


# In[23]:


print(model)


# In[24]:


import pandas as pd
import numpy as np

# Create input exactly as training schema
input_data = pd.DataFrame([{
    "supplier_tier": 2,
    "disruption_severity": 4,
    "production_impact_pct": 35.0,
    "response_time_days": 10,
    "log_revenue_loss": np.log1p(250000),  # VERY IMPORTANT
    "disruption_type": "Logistics Delay",
    "industry": "Manufacturing",
    "supplier_region": "Asia",
    "supplier_size": "Medium"
}])

prediction = model.predict(input_data)
print("Predicted Full Recovery Days:", prediction[0])


# In[ ]:




