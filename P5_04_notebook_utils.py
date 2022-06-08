""" Utils """
import os
import math
import warnings
import math
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from string import ascii_letters
from time import time
from matplotlib import pyplot as plt
from hyperopt import hp, fmin, tpe, hp, anneal, Trials, STATUS_OK, space_eval
from sklearn import preprocessing, decomposition, metrics
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
from yellowbrick.features import RadViz
from yellowbrick.datasets import load_concrete
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline                          
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, QuantileTransformer, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     cross_validate,
                                     StratifiedKFold,
                                     RepeatedStratifiedKFold,
                                     GridSearchCV,
                                     learning_curve
                                    )

#
init_notebook_mode(connected = True)
                              
def info(dataframe):
    """Prints dataframe parameters
    Args:
        dataframe (pd.Dataframe): data source
    Returns:
        -
    """
    print(str(len(dataframe.columns.values)) + " columns" )
    print(str(len(dataframe)) + " rows")
    print("Rate of missing values in df : " + str(dataframe.isnull().mean().mean()*100) + " %")
    
    
def inter_quartile_method_function(dataframe):
    '''IQ method applied on dataset
        Args:
            dataframe (pd.Dataframe): input
        Returns:
            dataframe (pd.Dataframe): output
    '''
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    iqr = q3 - q1
    dataframe = dataframe[(dataframe <= dataframe.quantile(0.75) + 1.5*iqr) 
                          & (dataframe >= dataframe.quantile(0.25) - 1.5*iqr)]
    
    return dataframe



def max_outliers_quantile_function(dataframe, quantile):
    '''General function for model creation and evaluation
        Args:
            dataframe (pd.Dataframe): input
            quantile (int): 
        Returns:
            dataframe (pd.Dataframe): output
    '''
    dataframe = dataframe[(dataframe <= dataframe.quantile(quantile))]
    
    return dataframe


def trim_by_period(dataframe, period, start_time):
    """
    Args:
        dataframe (pd.Dataframe): data source
        period (int): size of time period in days to consider for trimming
        start_time (int): start time in days out of the total time period of 695 days
    Returns:
        dataframe (pd.Dataframe): trimmed dataframe for the time period
    """
    max_time = dataframe.last_recency_rfm.max()
    min_time = dataframe.last_recency_rfm.min()
    end_time = max_time - min_time
    
    if start_time <= end_time :
        
        dataframe_period = dataframe[(dataframe.last_recency_rfm >= min_time + end_time - start_time) & (dataframe.last_recency_rfm <= min_time + end_time - start_time + period)]
        
        return dataframe_period

def simulation_function(dataframe, clustering_type, best_params, period, start_time):
    '''General function for model creation and evaluation
        Args:
            dataframe (pd.Dataframe): input
            numerical_transformers (list(str)): list of numerical features in X
            categorical_transformers (list(str)): list of categorical features in X
            clustering_type (sklearn function): type of clustering
            max_evals (int): number of hyperopt maximum evaluations
        Returns:
            -
    '''
    max_elapsed_time = dataframe.last_recency_rfm.max()
    min_elapsed_time = dataframe.last_recency_rfm.min()
    end_time = max_elapsed_time - min_elapsed_time
    
    X0 = trim_by_period(dataframe, period, start_time) 
    m0 = clustering_type(**best_params).fit(X0)
    labels_0 = m0.predict(X0)
    
    ari_list = []
    for i in range(start_time, end_time, period) :
       
        #if x_days_start <= end_time :
        X = trim_by_period(dataframe, period, i)
        m = clustering_type(**best_params).fit(X)

        labels = m.predict(X)
        labels_m0 = m0.predict(X)

        ari = adjusted_rand_score(labels, labels_m0)
        ari_list.append(ari)

    df_ari_plot = pd.DataFrame(ari_list)

    return df_ari_plot


def schema_function(numerical_variables_list, categorical_variables_list, clustering_type):
    '''Function for displaying pipleline schema
        Args:
            numerical_transformers (list(str)): list of numerical features in X
            categorical_transformers (list(str)): list of categorical features in X
            clustering_type (sklearn function): type of clustering
        Returns:
            Pipeline
    '''
    # 
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputation_median", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerical", numeric_preprocessor, numerical_variables_list),
            ("categorical", categorical_preprocessor, categorical_variables_list)
        ]
    )

    # Create pipeline
    pipe = make_pipeline(preprocessor, clustering_type)

    return pipe

        
class ClusteringBestParamsSearch:
                                      
    def __init__(self, dataframe,
                       numerical_transformers, 
                       categorical_transformers, 
                       clustering_type,
                       max_evals,
                       **kwargs):

        """
        General function for clustering model creation, optimization and evaluation
        Args:
            dataframe (pd.Dataframe): input
            numerical_transformers: steps for numerical features treatments
            categorical_transformers: steps for categorical features treatments
            clustering_type (sklearn function): name of clustering algorithm
            max_evals (int): number of hyperopt maximum evaluations
        Returns:
            -
        """ 
        self.dataframe = dataframe
        self.numerical_transformers = numerical_transformers
        self.categorical_transformers = categorical_transformers
        self.clustering_type = clustering_type
        self.max_evals = max_evals
        self.clustering_type_string = str(clustering_type)
        
    
    def define_pipeline(self, params):
    
        ##### Pipeline #####
        # Create sub-pipelines for numartical & categorical features
        numerical_preprocessor = make_pipeline(*self.numerical_transformers)
        categorical_preprocessor = make_pipeline(*self.categorical_transformers)
 
        # Associate both pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_preprocessor, make_column_selector(dtype_include="number")), 
                ("categorical", categorical_preprocessor, make_column_selector(dtype_include=["category"]))
            ]
        )
        
        # Create general pipeline
        model = Pipeline(steps=[("pre-processor", preprocessor),("clustering", self.clustering_type(**params))])
        return model
    
    
    def space_definition(self):

        if "KMeans" in self.clustering_type_string:
            # Define search space
            space = {'n_clusters': hp.choice('n_clusters', [2,3,4,5]),
                     'init': hp.choice('init', ['k-means++','random']),
                     'n_init': hp.choice('n_init', range(5, 15, 1)),
                     'max_iter': hp.choice('max_iter', range(100, 300, 20)),
                     #'tol': 1e-4,
                     #'verbose': 0,
                     'random_state': 1234,
                     #'copy_x': hp.choice('kmeans__copy_x',[True,False]),
                     'algorithm': hp.choice('algorithm', ["auto","full","elkan"])
                    }
                    

        elif "DBSCAN" in self.clustering_type_string:
            # Define search space
            space = {'eps': 0.5,
                     'min_samples': 5,
                     #'metric':,
                     #'metric_params':,
                     'algorithm': 'auto',
                     'leaf_size': 30,
                     #'p':
                     #'n_jobs':
                    }
            
        elif "GaussianMixture" in self.clustering_type_string:
            # Define search space
            space = {'n_components': hp.choice('n_components', [2,3,4]), 
                     'covariance_type': hp.choice('covariance_type', ['full','tied','diag','spherical']), 
                     'tol': 1e-3, 
                     'reg_covar': 1e-6,
                     'max_iter': hp.choice('max_iter', [90,100,110]),
                     'n_init': 1,
                     'init_params': 'kmeans',
                     #'weights_init': None
                     #'means_init': None
                     #'precisions_init': None
                     'random_state': 1234
                     #'warm_start': False
                     #'verbose': 0
                     #'verbose_interval': 10
                    }
                                     
        elif "Birch" in self.clustering_type_string:
            # Define search space
            space = {'threshold': hp.choice('threshold', [0.4,0.5,0.6]), 
                     'branching_factor': hp.choice('branching_factor', [40,50,70]),
                     'n_clusters': hp.choice('n_clusters', [3,4,5]),
                     #'compute_labels':True, 
                     #'copy':True
                    }
                    
        return space
    

    # Définition de la fonction objective 
    def tuning_objective(self, args): 

        pipe = self.define_pipeline(args)    
        pipe.fit(self.dataframe)

        if "GaussianMixture" in self.clustering_type_string:
            labels = pipe.predict(self.dataframe)
        else:
            labels = pipe.named_steps['clustering'].labels_
            
        scaled_dataframe = pipe.named_steps['pre-processor'].transform(self.dataframe) 
        
        silhouette = metrics.silhouette_score(scaled_dataframe, labels, metric='euclidean')
        print(f"silhouette: {silhouette}")
        print(args)
        
        return {"loss": -silhouette, "status": STATUS_OK}   
    
    
    def trials_clustering(self):
        
        # Initialize trials object
        trials = Trials()
        search_space = self.space_definition()
        print(search_space)
        print(self.max_evals)

        best = fmin(fn=self.tuning_objective,
                    space = search_space, 
                    algo=tpe.suggest, 
                    max_evals=self.max_evals, 
                    #trials=trials,
                    rstate=np.random.RandomState(1234)
                   )

        best_params = space_eval(search_space, best)
        print(best_params)
        return best_params
        
    def executes_algo_function(self, best_params):
        
        best_pipe = self.define_pipeline(best_params) 
        start_at = datetime.now()
        best_pipe.fit(self.dataframe)         
        end_at = datetime.now()
        running_time = end_at - start_at
        scaled_dataframe = best_pipe.named_steps['pre-processor'].transform(self.dataframe) 

        if "GaussianMixture" in self.clustering_type_string:
            labels = best_pipe.predict(self.dataframe)
        else:
            labels = best_pipe.named_steps['clustering'].labels_

        return labels, running_time, best_pipe, scaled_dataframe 
        

def plot_function(X, 
                  clustering_name, 
                  labels,
                  label_x,
                  label_y,
                  label_z):
    '''Plots data in 2D 
        Args:
            X (pd.Dataframe): input
            clustering_name (sklearn function): name of clustering algorithm
            label_x (str): name of x label
            label_y (str): name of y label
            label_z (str): name of z label
        Returns:
            -
    '''
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()

    if str(type(X)) == "<class 'numpy.ndarray'>":
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=1) 
    else:
        scatter = ax.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, alpha=1)  
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="center right", 
                        title="Clusters", 
                        fontsize='x-large', 
                        title_fontsize='x-large', 
                        borderpad=0.4,
                        framealpha=0.3,
                        facecolor="black",
                        edgecolor="black")

    
    #ax.add_artist(legend1)
    
    plt.title(clustering_name + " clustering")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    ax.grid(True)
    
    plt.show()
    
    
def compare_labels_function(X, 
                            clustering_name_list, 
                            output_algo_list,
                            label_x,
                            label_y,
                            label_z
                           ):
    '''Plots the 4 use case clusterings in 2D representation on a 2x2 grid
        Args:
            X (pd.Dataframe): input
            clustering_name_list (list(str)): list of the clustering algorithms
            output_algo_list (list(numpy.array(int))): list of the labels of the clustering algorithms
            label_x (str): name of x label
            label_y (str): name of y label
            label_z (str): name of z label
           
        Returns:
            -
    '''
    X = X.to_numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(20,20))
    
    k = 0
    for i in range(2):
        for j in range(2):
            scatter = axes[i][j].scatter(X[:,0], X[:,1], c=output_algo_list[k])
            
            # produce a legend with the unique colors from the scatter
            legend1 = axes[i][j].legend(*scatter.legend_elements(),
                                        loc="center right", 
                                        title="Clusters", 
                                        fontsize='x-large', 
                                        title_fontsize='x-large', 
                                        borderpad=0.4,
                                        framealpha=0.3,
                                        facecolor="black",
                                        edgecolor="black")
            axes[i][j].set_title(clustering_name_list[k] + " clustering")
            axes[i][j].set_xlabel(label_x)
            axes[i][j].set_ylabel(label_y)
            axes[i][j].grid(True)
            k = k +1
    plt.show()
    

def stats_function(X, labels, running_time):
    '''General function for model creation and evaluation
        Args:
            X (pd.Dataframe): input
            labels (numpy.array(int)): labels of the clustering algorithm
            running_time (): running time of the clustering algorithm
        Returns:
            -
    '''
    #
    silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    calinski_harabasz = metrics.calinski_harabasz_score(X, labels)
    davies_bouldin = metrics.davies_bouldin_score(X, labels)
    running_time = running_time.total_seconds()
    #
    print("")
    print('Silhouette Score: ' + str(silhouette))
    print('Calinski-Harabasz Score: ' + str(calinski_harabasz))
    print('Davies-Bouldin Score: ' + str(davies_bouldin))
    print('Running time: ' + str(running_time))
    print("")
    
    stats = [silhouette, calinski_harabasz, davies_bouldin, running_time]
    
    return stats


def sum_up_table_function(metric, order, **kwargs):
    """ 
    Stats sum up for the given algorithms 
    Args:
        metric (string): sklearn metric
        order (string): ascending or descending
    Returns:
        dataframe (pd.Dataframe): data output
    """                          
    # data
    d = {}
    
    for kwarg in kwargs:
        newline = {kwarg: kwargs[kwarg]}
        d.update(newline)

    index = ['Silhouette',
             'Calinski-Harabasz',
             'Davies-Bouldin Score',
             'Running time']

    # dataframe
    df = pd.DataFrame(data=d, index=index).transpose()

    order = order.lower()
    
    #
    if order == 'ascending':
        order = True 
    else: 
        order = False 
    
    #
    return df.sort_values(by=metric, axis=0, ascending=order, inplace=False, kind='quicksort')


def radar_chart_prepare_data_function(dataframe, algo_name, labels, use_case): 
    '''Agreggates an normalizes data by dividing each dataframe column by the biggest value in the given column
        Args:
            dataframe (pd.Dataframe): input
            algo_name (str): name of the algorithm
            labels (numpy.array(int)): labels of the clustering algorithm
            use_case (int): number of the use case
        Returns:
            dataframe (pd.Dataframe): output
    '''
    
    column_labels = 'labels_' + algo_name
    dataframe[column_labels] = labels
    
    if use_case  == 1 or use_case  == 2 :
        dataframe_0 = dataframe.sort_values(column_labels).groupby(column_labels).agg({dataframe.columns[0]: 'mean',
                                                                                   dataframe.columns[1]: 'mean',
                                                                                   dataframe.columns[2]: 'mean'})

    
    elif use_case  == 3 : 
        dataframe_0 = dataframe.groupby(column_labels).agg({dataframe.columns[0]: 'mean',
                                                               dataframe.columns[1]: 'mean',
                                                               dataframe.columns[2]: 'mean',
                                                               dataframe.columns[5]: 'mean'})
        
    elif use_case  == 4 : 
        dataframe_0 = dataframe.groupby(column_labels).agg({dataframe.columns[0]: 'mean',
                                                               dataframe.columns[1]: 'mean',
                                                               dataframe.columns[2]: 'mean',
                                                               dataframe.columns[6]: 'mean'})   
        
    
    dataframe_1 = dataframe_0.apply(lambda x: x / x.max())
    
    return dataframe_0, dataframe_1


def radar_chart_plot_function(dataframe, algo_name, labels, n_clusters, use_case):
    '''Plots the radar chart of the alogorithm
        Args:
            dataframe (pd.Dataframe): input
            algo_name (str): name of the algorithm
            labels (numpy.array(int)): labels of the clustering algorithm
            n_clusters (int): number of clusters of the clustering
            use_case (int): number of the use case

        Returns:
            -
    '''
    
    dataframe_normalized = radar_chart_prepare_data_function(dataframe, algo_name, labels, use_case)[1]
    categories = dataframe_normalized.columns    

    #
    unique, counts = np.unique(labels, return_counts=True)
    clusters_pop = dict(zip(unique, counts))
    
    fig = go.Figure()

    for i in range(n_clusters):

        fig.add_trace(go.Scatterpolar(
                                      r=dataframe_normalized.iloc[i].tolist(),
                                      theta=categories,
                                      fill='toself',
                                      name='Cluster ' + str(i) + ', ' + 'Pop: ' + str(counts[i])
                                     )
                     )

    fig.update_layout(polar=dict(radialaxis=dict(visible=True,
                                                 range=[0, 1],
                                                )),
                                                  showlegend=True,
                                                  width=600, 
                                                  height=600,
                                                  title="Clustering Radar Charts (Normalized) - " + algo_name + " Clustering",
                                                )

    fig.show()
    
            
def variables_to_cast_as_sring_function(dataframe, variables_to_cast_as_string_list):
    '''General function for model creation and evaluation
        Args:
            dataframe (pd.Dataframe): input
            variables_to_cast_as_string_list (list(str)): list of numerical features in X
        Returns:
            -
    '''
    # Cast non-computable numerical variable as strings
    dataframe[variables_to_cast_as_string_list] = dataframe[variables_to_cast_as_string_list].astype(str)

    return dataframe
    

def prepare_data(dataframe, numerical_list, categorical_list):
    '''General function for model creation and evaluation
        Args:
            dataframe (pd.Dataframe): input
            numerical_transformers (list(str)): list of numerical features in X
            categorical_transformers (list(str)): list of categorical features in X
        Returns:
            dataframe (pd.Dataframe): output
    '''
    
    X = dataframe.copy()
    
    X[numerical_list] = max_outliers_quantile_function(X[numerical_list], 0.99999)
    X[numerical_list] = preprocessing.StandardScaler().fit_transform(X[numerical_list])
    
    #
    if categorical_list :
        for col in X[categorical_list].columns:
            X[col] = preprocessing.LabelEncoder().fit_transform(X[col])
        
    X = X[numerical_list + categorical_list] 
    X = X.dropna(axis='rows', how='any')

    return X 


def pca_function(X, n_comp):
    '''PCA
        Args:
            X (pd.Dataframe): input
            n_comp (int): number of components of the PCA
        Returns:
            X_projected (numpy.array): Data after PCA teatment
    '''
    # Number of components to compute
    n_comp = n_comp

    # Preparing data for ACP
    names = X.index 
    features = X.columns

    # Mean centering and dimensionality reduction 
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)

    # Principal components computation
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(X_scaled)

    # Screen plot of eigenvalues
    display_scree_plot(pca)

    # Unity circle of correlations
    pcs = pca.components_
    display_circles(pcs, n_comp, pca, [(0,1),(2,3)], labels = np.array(features))

    # Individuals projection
    X_projected = pca.transform(X_scaled)
    #display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3)], labels = np.array(names))
    plt.show()
    
    print("explained variance ratio:" + str(pca.explained_variance_ratio_.cumsum()))
    print("")
    return X_projected


def pca_scaled_function(X, n_comp):
    '''Scaling data and PCA
        Args:
            X (pd.Dataframe): input
            n_comp (int): number of components of the PCA
        Returns:
            X_projected (numpy.array): Data after PCA teatment
    '''
    # Number of components to compute
    n_comp = n_comp

    # Preparing data for ACP
    names = X.index 
    features = X.columns

    # Principal components computation
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(X)

    # Screen plot of eigenvalues
    #display_scree_plot(pca)

    # Unity circle of correlations
    pcs = pca.components_
    #display_circles(pcs, n_comp, pca, [(0,1),(2,3)], labels = np.array(features))

    # Individuals projection
    X_projected = pca.transform(X)
    #display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3)], labels = np.array(names))
    plt.show()
    
    print("PCA explained variance ratio:" + str(pca.explained_variance_ratio_.cumsum()))
    
    return X_projected


def tsne_scaled_function(X):
    '''TSNE function
        Args:
            X (pd.Dataframe): input
        Returns:
            Y (pd.Dataframe): putput
    '''

    X_embedded = TSNE(n_components=3, init='random').fit_transform(X)
    
    return X_embedded

def plot_3d_function(X, algo_name, labels, label_x, label_y, label_z):
    '''General function for model creation and evaluation
        Args:
            X (pd.Dataframe): input
            algo_name (str): name of the algorithm
            clustering_name (sklearn function): name of clustering algorithm
            label_x (str): name of x label
            label_y (str): name of y label
            label_z (str): name of z label
        Returns:
            -
    '''
    
    if label_x == "count_frequency_rfm":
        X = X.to_numpy()
    #
    fig = px.scatter_3d(X, 
                        x=0,
                        y=1,
                        z=2,
                        title='',
                        labels={'0': label_x, '1': label_y, '2': label_z},
                        color=labels
                       )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """ Plots PCA circle of correlation

    Args:
        pcs (numpy.ndarray):
        n_comp (int):
        pca (sklearn.decomposition._pca.PCA):
        axis_ranks (list):

    Returns:
        Plot
    """
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:],
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x,
                                 y,
                                 labels[i],
                                 fontsize='14',
                                 ha='center',
                                 va='center',
                                 rotation=label_rotation,
                                 color="blue",
                                 alpha=0.5
                                )

            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    """ Plots PCA factorial planes

    Args:
        X_projected (numpy.ndarray):
        n_comp (int): number of components to compute
        pca (sklearn.decomposition._pca.PCA):
        axis_ranks (list):

    Returns:
        Plot

    """
    for d1,d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            plt.rc('axes', unicode_minus=False)
            fig = plt.figure(figsize=(7,6))


            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1],
                                X_projected[selected, d2],
                                alpha=alpha,
                                label=value
                               )
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i], fontsize='14', ha='center',va='center')

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


def display_scree_plot(pca):
    """ Plots PCA scree plot
    Args:
        pca (sklearn.decomposition._pca.PCA):

    Returns:
        Plot
    """
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(7,6))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)





