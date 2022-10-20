import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from time import time
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



def read_prep_data(file):
    df = pd.read_csv(r"%s" % file, sep=';')
    # missining selected
    df = df[df['Selected'] != 'missing']  # make sure that we have data for selected attribute

    # CreditScore handeling
    df.CreditScore.replace(0.0, np.nan, inplace=True)
    df['CreditScore'].fillna((df['CreditScore'].median()), inplace=True)

    # MonthlyCost handeling
    df.MonthlyCost.replace(0.0, np.nan, inplace=True)
    df['MonthlyCost'].fillna((df['MonthlyCost'].median()), inplace=True)

    # NumberOfTerms handeling
    df.NumberOfTerms.replace(0.0, np.nan, inplace=True)
    df['NumberOfTerms'].fillna((df['NumberOfTerms'].median()), inplace=True)

    # FirstWithdrawalAmount handeling
    df.FirstWithdrawalAmount.replace(0.0, np.nan, inplace=True)
    df['FirstWithdrawalAmount'].fillna((df['FirstWithdrawalAmount'].median()), inplace=True)

    # map Selected to 1(signed), and 0(not signed)
    df['Selected'] = df['Selected'].map({'True': 1, 'False': 0})

    # for FrequencyOfIncompleteness
    #     df2 = df[df['Activity']=='A_Incomplete']
    #     df_new = pd.DataFrame(df2.groupby(['Case ID'])['Activity'].count()).reset_index()
    #     df_new.columns = ['Case ID', 'FrequencyOfIncompleteness']
    #     df_new = pd.DataFrame(df_new.groupby('Case ID')['FrequencyOfIncompleteness'].sum()).reset_index()
    #     df = pd.merge(df_new, df, on='Case ID')

    # For NumberOfOffers
    df2 = df[df['Activity'] == "O_Created"]  # to count offers
    df_new = pd.DataFrame(df2.groupby(['Case ID'])['Activity'].count()).reset_index()
    df_new.columns = ['Case ID', 'NumberOfOffers']
    df = pd.merge(df_new, df, on='Case ID')

    # For matchRequested
    df['MatchedRequest'] = np.where((df.RequestedAmount <= df.OfferedAmount), 'True', 'False')

    df = df.groupby('Case ID').apply(get_duration)
    df = df.reset_index(drop=True)

    # O_Sent (mail and online)
    #     df2 = df[df['Activity'] == 'O_Sent (mail and online)'] # to count offers
    #     df_new = pd.DataFrame(df2.groupby(['Case ID'])['Activity'].count()).reset_index()
    #     df_new.columns = ['Case ID', 'O_sent_mail_online_frequency']
    #     df = pd.merge(df_new, df, on='Case ID')

    # O_Sent (online only)
    #     df2 = df[df['Activity'] == 'O_Sent (online only)'] # to count offers
    #     df_new = pd.DataFrame(df2.groupby(['Case ID'])['Activity'].count()).reset_index()
    #     df_new.columns = ['Case ID', 'O_sent_online_only_frequency']
    #     df = pd.merge(df_new, df, on='Case ID')

    # binning columns
    #    df['new_duration'] = pd.cut(df['durationDays'], [0,8,15,30,31,168],
    #                                include_lowest=True, right=False, labels=['0-7','8-14','15-29','30','31+'])
    #    df['new_duration'] = df['new_duration'].astype(str)
    #    df['new_FrequencyOfIncompleteness'] = pd.cut(df['FrequencyOfIncompleteness'], 15)
    #    df['new_FrequencyOfIncompleteness'] = df['new_FrequencyOfIncompleteness'].astype(str)xs

    df['binned_RequestedAmount'] = pd.qcut(df['RequestedAmount'], 5, labels=['0-5000', '5001-10000',
                                                                             '10001-15000', '15001-25000', '25000+'])
    df['binned_RequestedAmount'] = df['binned_RequestedAmount'].astype(str)

    df['binned_duration'] = pd.qcut(df['durationDays'], 5, labels=['0-8', '9-13', '14-22', '23-30', '30+'])
    df['binned_duration'] = df['binned_duration'].astype(str)

    df['binned_NoOfTerms'] = pd.qcut(df['NumberOfTerms'], 5, labels=['6-48', '49-60', '61-96', '97-120', '120+'])
    df['binned_NoOfTerms'] = df['binned_NoOfTerms'].astype(str)

    df['binned_CreditScore'] = pd.qcut(df['CreditScore'], 2, labels=['low', 'high'])
    df['binned_CreditScore'] = df['binned_CreditScore'].astype(str)

    df['binned_MonthlyCost'] = pd.qcut(df['MonthlyCost'], 5, labels=['40-148', '149-200', '201-270',
                                                                     '271-388', '388+'])
    df['binned_MonthlyCost'] = df['binned_MonthlyCost'].astype(str)

    df['binned_FirstWithdrawalAmount'] = pd.qcut(df['FirstWithdrawalAmount'], 3,
                                                 labels=['0-7499', '7500-9895', '9896-75000'])
    df['binned_FirstWithdrawalAmount'] = df['binned_FirstWithdrawalAmount'].astype(str)

    df['binned_NumberOfOffers'] = pd.cut(df['NumberOfOffers'], [1, 2, 3, 11],
                                         include_lowest=True, right=False, labels=['1', '2', '3+'])
    df['binned_NumberOfOffers'] = df['binned_NumberOfOffers'].astype(str)

    df = df.groupby('Case ID').apply(keep_last)
    df = df.reset_index(drop=True)

    # lower case
    column = ['ApplicationType', 'LoanGoal', 'MatchedRequest']
    for col in column:
        df[col] = df[col].str.lower()

    return df

def keep_last(group):
    return group.tail(1)


def get_duration(gr):
    df = pd.DataFrame(gr)
    if len(df[(df["Activity"] == "A_Denied") | (df["Activity"] == "A_Cancelled") | (
            df["Activity"] == "A_Pending")]) > 0:
        df['new_date'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df['time:timestamp']]

        first_dt = df[df['Activity'] == 'O_Create Offer']['new_date']
        last_dt = \
            df[(df["Activity"] == "A_Denied") | (df["Activity"] == "A_Cancelled") | (df["Activity"] == "A_Pending")][
                'new_date']

        first_dt = first_dt[first_dt.index.values[0]]
        # print(last_dt)
        last_dt = last_dt[last_dt.index.values[0]]

        d1 = parse(str(first_dt))
        d2 = parse(str(last_dt))

        delta_days = (d2 - d1).days
        # print(delta_days,'\n')
        df['durationDays'] = delta_days
        return df


data = 'BPIC17_O_Accepted.csv'
feat_eng_data = read_prep_data(data)


le = preprocessing.LabelEncoder()
loan_df = pd.read_csv('BPIC17_O_Accepted.csv',sep=";")

loan_df.head(10)
loan_df = loan_df[loan_df.Selected!='missing']

loan_df['Selected'].value_counts()

object_cols = ['ApplicationType','label','LoanGoal','Activity','Selected']
for col in object_cols:
    loan_df[col] = le.fit_transform(loan_df[col])


loan_df_grouped = loan_df.groupby(['Case ID'])

loan_numeric_df = loan_df[['ApplicationType','LoanGoal','RequestedAmount','Activity','Selected','CreditScore']]

loan_numeric_df = loan_numeric_df.iloc[0:1000,]


loan_numeric_df['Selected'].value_counts()



from causalnex.structure.notears import from_pandas

sm = from_pandas(loan_numeric_df)


sm.edges


from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

viz = plot_structure(
    sm,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
    prog='fdp',
)
Image(viz.draw(format='png'))



sm.remove_edges_below_threshold(0.8)
viz = plot_structure(
    sm,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
Image(viz.draw(format='png'))



import sensemakr as smkr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
from datetime import datetime
from dateutil.parser import parse
from statsmodels.iolib.summary2 import summary_col

reg_model2 = smf.ols(formula='Selected ~ ApplicationType + CreditScore + LoanGoal + Activity + RequestedAmount',data=loan_numeric_df)
                     
loan_model = reg_model2.fit() 


summary_col(loan_model, regressor_order=["RequestedAmount"], drop_omitted=True)

loan_sense =  smkr.Sensemakr(model = loan_model,
                              treatment = "RequestedAmount",
                              benchmark_covariates = ["CreditScore"],
                              kd = [1,2,3],
                              ky = [1,2,3],
                              q = 1.0,
                              alpha = 0.05,
                              reduce = True)


html_code = loan_sense.ovb_minimal_reporting(format = "html")