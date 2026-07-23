#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shap
import lime
import lime.lime_tabular
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[2]:


# 1. Load Data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


# 2. Train Model (This makes it NOT a pretrained model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[11]:


# --- SHAP SECTION ---
# TreeExplainer is specific to models like Random Forest
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap(X_test)


# In[12]:


# Plot: SHAP Heatmap (Visualizing the "Shape" of the data influence)
plt.figure()
shap.plots.heatmap(shap_values[:,:,1]) # For class 1
plt.show()


# In[14]:


plt.figure()


# In[16]:


shap_values_obj = explainer_shap(X_test)


# In[17]:


shap.summary_plot(list(shap_values_obj.values.transpose(2,0,1)), X_test, plot_type="bar", class_names=iris.target_names)
plt.show()


# In[20]:


# --- 4. LIME SECTION ---
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=iris.target_names,
    mode='classification'
)


# In[21]:


exp = explainer_lime.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp.as_pyplot_figure()
plt.tight_layout()
plt.show()


# In[22]:


#predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns))


# In[ ]:





# In[ ]:




