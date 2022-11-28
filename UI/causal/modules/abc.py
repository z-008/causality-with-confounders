import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from time import time
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import EditableGraph
from causalnex.structure.notears import from_pandas
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure.notears import from_pandas
from causalnex.network import BayesianNetwork
from causalnex.discretiser import Discretiser
from causalnex.structure import StructureModel
from typing import List, Tuple
from causalnex.plots import plot_structure
from IPython.display import Image
import dataframe_image
import networkx as nx
import os
import sensemakr as smkr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
from datetime import datetime
from dateutil.parser import parse
from statsmodels.iolib.summary2 import summary_col


def causal_graph(target,feature_lst,benchmark,treatment):

    loan_df = pd.read_csv('causal/clean/clean.csv')
    le = preprocessing.LabelEncoder()
    # object_cols = []
    # loan_df = loan_df[loan_df.Selected!='missing']


    for col in feature_lst:
        if isinstance(col,(str, bool)):
           loan_df[col] = le.fit_transform(loan_df[col])
    # object_cols = ['ApplicationType','label','LoanGoal','Activity','Selected']
    # for col in object_cols:
    #     loan_df[col] = le.fit_transform(loan_df[col])

    # loan_df_grouped = loan_df.groupby(['Case ID'])
   
    loan_numeric_df = loan_df[feature_lst]

    # loan_numeric_df = loan_numeric_df.iloc[0:1000,]

    sm = from_pandas(loan_numeric_df)

    sm.edges

    viz = plot_structure(
        sm,
        graph_attributes={"scale": "0.5"},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK,
        prog='fdp',
    )
# Image(viz.draw(format='png'))
    
    sm.remove_edges_below_threshold(0.8)
    viz = plot_structure(
        sm,
        graph_attributes={"scale": "0.5"},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK,
    )
# Image(viz.draw(format='png'))

# edge_color = dict()
# for  edge in enumerate(g.edges):
#     edge_color[edge] = 'tab:gray' 

# node_color = dict()
# for node in g.nodes:
#     node_color[node] = 'tab:red' 

    g = nx.Graph(viz)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_instance = EditableGraph(
        g, node_size=5, node_color='tab:red',
        node_labels=True, node_label_offset=0.1, node_label_fontdict=dict(size=20),
        edge_color='tab:gray', edge_width=2,
        arrows=True, ax=ax)

    # plt.show()
    plt.savefig("causal/static/img/graph.png")

    context = {}
    graph_nodes = plot_instance.nodes
    graph_edges = plot_instance.edges

    context['nodes'] = graph_nodes
    edges = []
    for edge in graph_edges:
        node_1, node_2 = edge
        if [node_2, node_1, "T", "F"] in edges:
            print('Already in list')
            edges.remove([node_2, node_1, "T", "F"])
            edge_list = []
            edge_list.append(node_1)
            edge_list.append(node_2)
            edge_list.append("T")
            edge_list.append("T")
        else:
            edge_list = []
            edge_list.append(node_1)
            edge_list.append(node_2)
            edge_list.append("T")
            edge_list.append("F")
            edges.append(edge_list)
    context['edges'] = edges
    
    feature_lst.remove(target)
    formla = target + " ~ "
    for col in feature_lst:
            if formla[-2]=='~':
                formla = formla + col
            else:
                formla = formla +" + "+ col
    
    reg_model2 = smf.ols(formula=formla,data=loan_numeric_df)
                     
    loan_model = reg_model2.fit() 


    summary_col(loan_model, regressor_order=[treatment], drop_omitted=True)

    loan_sense =  smkr.Sensemakr(model = loan_model,
                              treatment = treatment,
                              benchmark_covariates = [benchmark],
                              kd = [1,2,3],
                              ky = [1,2,3],
                              q = 1.0,
                              alpha = 0.05,
                              reduce = True)

    s = smkr.sensitivity_stats(model = loan_model,
                              treatment = treatment,q = 1.0,
                              alpha = 0.05,
                              reduce = True)

    r2yz_dx = loan_sense.bounds['r2yz_dx'][0]*100
    r2dz_x = loan_sense.bounds['r2dz_x'][0]*100

    html_code = loan_sense.ovb_minimal_reporting(format = "html")
    html_code = html_code.replace('\t','')
    html_code = html_code.replace('\n','')
    return context,html_code,s,r2yz_dx,r2dz_x