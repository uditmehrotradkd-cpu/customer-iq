import plotly.express as px

def pca_scatter(df):
    if 'PCA3' in df:
        return px.scatter_3d(df,x='PCA1',y='PCA2',z='PCA3',color='Persona',hover_data=['CustomerID','Recency','Frequency','Monetary'],title='3D Customer Segmentation')
    return px.scatter(df,x='PCA1',y='PCA2',color='Persona',hover_data=['CustomerID','Recency','Frequency','Monetary'],title='2D Customer Segmentation')

def segment_box(df, feature):
    return px.box(df,x='Persona',y=feature,color='Persona',points=False,title=f'{feature} by Segment')

def k_diagnostics(metrics):
    return (px.line(metrics,x='K',y='Inertia',markers=True,title='Elbow Curve'),
            px.line(metrics,x='K',y='Silhouette',markers=True,title='Silhouette Score'))
