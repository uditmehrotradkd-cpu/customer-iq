from pathlib import Path
import json
import pandas as pd
from .config import Config
from .data import clean_dataset, detect_schema, build_feature_table
from .preprocess import FeaturePreprocessor
from .model import evaluate_k, choose_k, fit_kmeans, fit_pca, save_artifact
from .personas import build_personas

def train_dataframe(raw, source_name='dataset', cfg=None):
    cfg=cfg or Config(); cfg.ensure_dirs()
    clean,quality=clean_dataset(raw); schema=detect_schema(clean,cfg); features=build_feature_table(clean,schema,cfg)
    model_columns=[c for c in features.columns if c!='CustomerID' and pd.api.types.is_numeric_dtype(features[c])]
    for c in ['Recency','Frequency','Monetary']:
        if c not in model_columns: raise ValueError(f'Required feature missing: {c}')
    medians={c:float(features[c].median()) for c in model_columns}
    prep=FeaturePreprocessor(model_columns); X=prep.fit_transform(features)
    metrics=evaluate_k(X,cfg.min_k,cfg.max_k,cfg.random_state); k=choose_k(metrics)
    model,labels=fit_kmeans(X,k,cfg.random_state); pca=fit_pca(X); coords=pca.transform(X)
    clustered=features.copy(); clustered['Cluster']=labels
    personas=build_personas(clustered); clustered['Persona']=clustered['Cluster'].map(personas)
    clustered['PCA1']=coords[:,0]
    if coords.shape[1]>1: clustered['PCA2']=coords[:,1]
    if coords.shape[1]>2: clustered['PCA3']=coords[:,2]
    artifact={'model':model,'preprocessor':prep,'pca':pca,'model_columns':model_columns,'personas':{int(k):v for k,v in personas.items()},'schema':schema,'quality':quality,'metrics':metrics,'n_clusters':k,'training_rows':len(clustered),'training_feature_medians':medians,'source_name':source_name}
    save_artifact(cfg.artifact_dir/'customer_iq_model.joblib',artifact)
    clustered.to_csv(cfg.output_dir/'customer_segments.csv',index=False); metrics.to_csv(cfg.output_dir/'cluster_metrics.csv',index=False)
    (cfg.output_dir/'training_report.json').write_text(json.dumps({'source':source_name,'rows':len(clustered),'features':model_columns,'selected_k':k,'silhouette':float(metrics.loc[metrics.K==k,'Silhouette'].iloc[0]),'quality':quality},indent=2),encoding='utf-8')
    return artifact,clustered,metrics
