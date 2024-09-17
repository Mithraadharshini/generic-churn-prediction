# src/explanations.py
import shap
from lime.lime_tabular import LimeTabularExplainer

def explain_global(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap.summary_plot(shap_values, X)

def explain_local(model, X, instance):
    explainer = LimeTabularExplainer(X, mode='classification')
    explanation = explainer.explain_instance(instance, model.predict_proba)
    return explanation.as_list()
