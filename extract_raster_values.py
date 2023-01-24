## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Extract mean values from various raster data per field

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import time
import os
import geopandas as gpd
import pandas as pd
import glob

from sklearn.linear_model import LogisticRegression as lr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics


from prop_match_functions import *
import math
import numpy as np
# import scipy.stats as stats
import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns


## project processing library
import processing_lib

WD = r"Q:\FORLand\Clemens\chapter02"
os.chdir(WD)

def merge_dataframes():
    fields = gpd.read_file(r"Q:\FORLand\Clemens\chapter02\data\vector\farms\IACS_BB_2020_with_owners.shp")
    fields = fields[
        ["BTNR", "GROESSE", "ID", "Oeko", "ID_KTYP", "ID_WiSo", "ID_HaBl", "CODE_neu", "certain_ow", "mother_com",
         "num_comp_c", "energy", "geometry"]]
    farm_sizes = fields.groupby("BTNR").agg(
        farm_area=pd.NamedAgg("GROESSE", "sum")
    ).reset_index()
    fields = pd.merge(fields, farm_sizes, "left", "BTNR")

    df_lst = glob.glob(r"Q:\FORLand\Clemens\chapter02\data\tables\extract_values\*.csv")
    df_lst = [pd.read_csv(pth, sep=";") for pth in df_lst]

    husbandry = pd.read_excel(
        r"Q:\FORLand\Daten\vector\InVekos\Brandenburg\invekos_2019_2020\zusaetzliche_infos_2020\2020_Anz_Rinder_Schweine_Schafe_Gefluegel.xlsx",
        dtype={"BNRZD": str}
    )
    fields = pd.merge(fields, husbandry, how="left", left_on="BTNR", right_on="BNRZD")
    for col in ["Rinder", "Schweine", "Schafe", "Gefluegel"]:
        fields.loc[fields[col].isna(), col] = 0

    for col in ["certain_ow", "mother_com", "num_comp_c"]:
        fields.loc[fields[col].isna(), col] = "unkown"

    fields.loc[fields["energy"].isna(), "energy"] = 999
    fields.loc[fields["ID_KTYP"].isna(), "ID_KTYP"] = 80

    for col in ["ID_KTYP", "ID_WiSo", "ID_HaBl"]:
        fields.loc[fields[col].isna(), col] = 99

    for df in df_lst:
        fields = pd.merge(fields, df, "left", "ID")

    fields.drop(columns=["geometry", "BNRZD"], inplace=True)
    fields.to_csv(r"Q:\FORLand\Clemens\chapter02\data\tables\all_fields_with_values_2020.csv", index=False)


def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 800)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', 800)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def data_exploration(df, c_factor_col, out_folder, plt_ind, year):


    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Field size vs. {c_factor_col}')
    sns.scatterplot(data=df, x="GROESSE", y=c_factor_col, hue="treatment", ax=ax)
    plt.close()
    fig.savefig(rf"{out_folder}\{plt_ind:02d}-field_size_vs_{c_factor_col}.png")

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Field size vs. {c_factor_col}')
    sns.boxplot(data=df, x="field_size_class", y=c_factor_col, hue="treatment", ax=ax)
    plt.close()
    fig.savefig(rf"{out_folder}\{plt_ind:02d}-field_size_vs_{c_factor_col}.png")

    ## Plot the mean c-factor vs farm size
    df_plt = df.groupby("BTNR").agg(
        mean_c_factor=pd.NamedAgg(c_factor_col, "mean"),
        farm_size=pd.NamedAgg("farm_area", "first"),
        treatment=pd.NamedAgg("treatment", "first"),
        farm_size_class=pd.NamedAgg("treatment", "first")
    )

    plt_ind += 1
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Probability density - {c_factor_col}')
    sns.displot(data=df_plt, x="mean_c_factor", stat="density", ax=ax)
    plt.savefig(rf"{out_folder}\{plt_ind:02d}-density_{c_factor_col}_of_farms.png")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Farm size vs. mean c-factor')
    sns.scatterplot(data=df_plt, x="farm_size", y="mean_c_factor", hue="treatment", ax=ax)
    fig.savefig(rf"{out_folder}\{plt_ind:02d}-farm_size_vs_mean_{c_factor_col}.png")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Farm size vs. mean c-factor')
    sns.scatterplot(data=df_plt.loc[df_plt["treatment"] != "unkown"], x="farm_size", y="mean_c_factor", hue="treatment",
                    ax=ax)
    fig.savefig(rf"{out_folder}\{plt_ind:02d}-farm_size_vs_mean_{c_factor_col}_wo_unkown.png")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Mean c-factor of farm size classes')
    sns.kdeplot(data=df_plt, x="mean_c_factor", hue="farm_size_class", ax=ax)
    fig.savefig(rf"{out_folder}\{plt_ind:02d}-kde_mean_{c_factor_col}_of_farm_size_class.png")
    plt.close()

    # print("\tPlot distribution and relations of data.")
    # t = df.groupby('treatment').apply(lambda x: x.sample(100)).reset_index(drop=True)
    # sns.pairplot(data=t, hue='treatment')

    txt_pth = f"{out_folder}\data_exploration.txt"
    with open(txt_pth, 'w') as f:
        f.write(f"##### Data exploration - {c_factor_col}-{year} #####\n")
    f.close()
    with open(txt_pth, 'a') as f:
        f.write("\nFarm area\n")
        df_temp = df.groupby('treatment').agg(mean_farm_size=pd.NamedAgg("farm_area", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nField size\n")
        df_temp = df.groupby('treatment').agg(mean_field_size=pd.NamedAgg("GROESSE", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nAckerzahl\n")
        df_temp = df.groupby('treatment').agg(mean_ackerzahl=pd.NamedAgg("ackerzahl", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nDEM\n")
        df_temp = df.groupby('treatment').agg(mean_dem=pd.NamedAgg("dem", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nSlope\n")
        df_temp = df.groupby('treatment').agg(mean_slope=pd.NamedAgg("slope", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nC-Factor\n")
        df_temp = df.groupby('treatment').agg(mean_c_factor=pd.NamedAgg(c_factor_col, "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nCattle\n")
        df_temp = df.drop_duplicates(subset="BTNR").groupby('treatment').agg(
            mean_num_cattle=pd.NamedAgg("Rinder", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nHogs\n")
        df_temp = df.drop_duplicates(subset="BTNR").groupby('treatment').agg(
            mean_num_hogs=pd.NamedAgg("Schweine", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nSheep\n")
        df_temp = df.drop_duplicates(subset="BTNR").groupby('treatment').agg(
            mean_num_sheep=pd.NamedAgg("Schafe", "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nCrop classes\n")
        t = df.groupby(['treatment', 'ID_KTYP']).agg(count_KTYP=pd.NamedAgg("ID_KTYP", "count")).reset_index()
        t["share"] = 100 * t['count_KTYP'] / t.groupby('treatment')['count_KTYP'].transform('sum')
        df_temp = pd.pivot_table(t[['treatment', 'ID_KTYP', 'share']], values='share', index='treatment',
                                 columns='ID_KTYP', aggfunc=np.mean)
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nSoil types\n")
        t = df.groupby(['treatment', 'soil_type']).agg(
            count_soil_type=pd.NamedAgg("soil_type", "count")).reset_index()
        t["share"] = 100 * t['count_soil_type'] / t.groupby('treatment')['count_soil_type'].transform('sum')
        df_temp = pd.pivot_table(t[['treatment', 'soil_type', 'share']], values='share', index='treatment',
                                 columns='soil_type', aggfunc=np.mean)
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nOeko\n")
        t = df.groupby(['treatment', 'Oeko']).agg(count_Oeko=pd.NamedAgg("Oeko", "count")).reset_index()
        t["share"] = 100 * t['count_Oeko'] / t.groupby('treatment')['count_Oeko'].transform('sum')
        df_temp = pd.pivot_table(t[['treatment', 'Oeko', 'share']], values='share', index='treatment',
                                 columns='Oeko', aggfunc=np.mean)
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nEnergy\n")
        t = df.groupby(['treatment', 'energy']).agg(count_energy=pd.NamedAgg("energy", "count")).reset_index()
        t["share"] = 100 * t['count_energy'] / t.groupby('treatment')['count_energy'].transform('sum')
        df_temp = pd.pivot_table(t[['treatment', 'energy', 'share']], values='share', index='treatment',
                                 columns='energy', aggfunc=np.mean)
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nFarm size class\n")
        t = df.groupby(['treatment', 'farm_size_class']).agg(
            count_farm_size_class=pd.NamedAgg("farm_size_class", "count")).reset_index()
        t["share"] = 100 * t['count_farm_size_class'] / t.groupby('treatment')['count_farm_size_class'].transform(
            'sum')
        df_temp = pd.pivot_table(t[['treatment', 'farm_size_class', 'share']], values='share', index='treatment',
                                 columns='farm_size_class', aggfunc=np.mean)
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nC-Factor by energy-col\n")
        df_temp = df.groupby('energy').agg(mean_c_factor=pd.NamedAgg(c_factor_col, "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nC-Factor by Oeko-col\n")
        df_temp = df.groupby('Oeko').agg(mean_c_factor=pd.NamedAgg(c_factor_col, "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nC-Factor by farm_size_class-col\n")
        df_temp = df.groupby('farm_size_class').agg(mean_c_factor=pd.NamedAgg(c_factor_col, "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nC-Factor by field_size_class-col\n")
        df_temp = df.groupby('field_size_class').agg(mean_c_factor=pd.NamedAgg(c_factor_col, "mean")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
    f.close()

    return plt_ind


def prop_match_wrapper(df, c_factor_col, out_folder, excl_attr, treatment_var_dict, plt_ind, year):
    ##################### MATCHING #####################
    txt_pth = f"{out_folder}\matching.txt"
    with open(txt_pth, 'w') as f:
        f.write(f"##### Matching -{c_factor_col}-{year} #####\n")
    f.close()

    ## Select two groups (e.g. small and large company networks), and prepare a reduced df with only necessary
    ## co-variates and treatment variable as binary dtype
    ## Column names: "BTNR", "ID", "ID_KTYP", "GROESSE", "Oeko", "num_comp_c", "energy", "farm_area", "Rinder",
    ## "Schweine", "Schafe", "ackerzahl", "dem", c_factor_col, "slope", "soil_type", "farm_size_class",
    ## "field_size_class"

    df = df.loc[(df["treatment"] != excl_attr) & (df["treatment"].notna()) & (df["treatment"] != "unkown")].copy()

    df.index = range(len(df))
    y = df[[c_factor_col]]
    df_data = df.copy()

    plt_ind += 1
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Probability density - {c_factor_col}')
    sns.displot(data=df_data, x=c_factor_col, stat="density", ax=ax)
    plt.savefig(rf"{out_folder}\{plt_ind:02d}-density_{c_factor_col}_of_fields.png")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Field size vs. {c_factor_col}')
    sns.scatterplot(data=df_data, x="GROESSE", y=c_factor_col,
                    hue="treatment", ax=ax)
    plt.close()
    fig.savefig(rf"{out_folder}\{plt_ind:02d}-field_size_vs_mean_{c_factor_col}_wo_unkown.png")

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'{c_factor_col} of field size classes')
    sns.kdeplot(data=df_data, x=c_factor_col, hue="field_size_class", ax=ax)
    plt.close()
    fig.savefig(rf"{out_folder}\{plt_ind:02d}-kde_{c_factor_col}_of_field_size_class.png")

    with open(txt_pth, 'a') as f:
        f.write("\nFarm area\n")
        df_temp = df_data.groupby('treatment').agg(count_fields=pd.NamedAgg("ID", "count")).reset_index()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
    f.close()
    df_data = df_data[["GROESSE", "Oeko", "treatment", "energy", "farm_area", "Rinder", "Schweine", "Schafe",
                       "ackerzahl", "dem", "slope", "soil_type"]].copy()
    df_data["treatment"] = df_data["treatment"].map(treatment_var_dict)

    ## Separate treatment from other variables
    T = df_data.treatment
    X = df_data.loc[:, df_data.columns != 'treatment']

    ## Convert categorical variables into dummy variables
    X_encoded = pd.get_dummies(
        data=X,
        columns=["Oeko", "energy", "soil_type"],
        prefix={'Oeko': 'eco', "energy": "energy", "soil_type": "soil"}, drop_first=False)

    ## Design pipeline to build the treatment estimator. It standardizes the data and applies a logistic classifier
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_classifier', lr())
    ])

    ## Fit the classifier to the data
    pipe.fit(X_encoded, T)
    predictions = pipe.predict_proba(X_encoded)
    predictions_binary = pipe.predict(X_encoded)

    ## Get some accuracy measurements
    with open(txt_pth, 'a') as f:
        f.write('\nAccuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
        f.write('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
        f.write('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))
    f.close()
    predictions_logit = np.array([logit(xi) for xi in predictions[:, 1]])

    plt_ind += 1
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
    sns.kdeplot(x=predictions[:, 1], hue=T, ax=ax[0])
    ax[0].set_title('Propensity Score')
    sns.kdeplot(x=predictions_logit, hue=T, ax=ax[1])
    ax[1].axvline(-0.4, ls='--')
    ax[1].set_title('Logit of Propensity Score')
    # plt.show()
    plt.savefig(rf"{out_folder}\{plt_ind:02d}-density-distr-propensity-score+logit-propensity-score.png")
    plt.close()

    common_support = (predictions_logit > -10) & (predictions_logit < 10)
    df_data.loc[:, 'propensity_score'] = predictions[:, 1]
    df_data.loc[:, 'propensity_score_logit'] = predictions_logit
    df_data.loc[:, 'outcome'] = y[c_factor_col]

    X_encoded.loc[:, 'propensity_score'] = predictions[:, 1]
    X_encoded.loc[:, 'propensity_score_logit'] = predictions_logit
    X_encoded.loc[:, 'outcome'] = y[c_factor_col]
    X_encoded.loc[:, 'treatment'] = df_data.treatment

    ## Use Nearerst Neighbors to identify matching candidates.
    ## Then perform 1-to-1 matching by isolating/identifying groups of (T=1,T=0).
    caliper = np.std(df_data.propensity_score) * 0.25

    with open(txt_pth, 'a') as f:
        f.write('\nCaliper (radius) is: {:.4f}\n'.format(caliper))
    f.close()

    df_data = X_encoded

    ## caliper reduces the space from which the neighbors are searched
    ## p defines how the distance is calculated. P=2 --> euclidean distance
    knn = NearestNeighbors(n_neighbors=10, p=2, radius=caliper)
    knn.fit(df_data[['propensity_score_logit']].to_numpy())

    distances, indexes = knn.kneighbors(
        df_data[['propensity_score_logit']].to_numpy(), n_neighbors=10)

    with open(txt_pth, 'a') as f:
        f.write('\nFor item 0, the 4 closest distances are (first item is self):')
        for ds in distances[0, 0:4]:
            f.write('\nElement distance: {:4f}'.format(ds))
        f.write('\n...')
        f.write('\nFor item 0, the 4 closest indexes are (first item is self):')
        for idx in indexes[0, 0:4]:
            f.write('\nElement index: {}'.format(idx))
        f.write('\n...')
    f.close()

    def perfom_matching_v2(row, indexes, df_data):
        current_index = int(row['index'])  # Obtain value from index-named column, not the actual DF index.
        prop_score_logit = row['propensity_score_logit']
        for idx in indexes[current_index, :]:
            if (current_index != idx) and (row.treatment == 1) and (df_data.loc[idx].treatment == 0):
                return int(idx)

    df_data['matched_element'] = df_data.reset_index().apply(perfom_matching_v2, axis=1, args=(indexes, df_data))
    treated_with_match = ~df_data.matched_element.isna()
    treated_matched_data = df_data[treated_with_match][df_data.columns]

    ## for all untreated matched observations, retrieve the co-variates
    def obtain_match_details(row, all_data, attribute):
        return all_data.loc[row.matched_element][attribute]

    untreated_matched_data = pd.DataFrame(data=treated_matched_data.matched_element)

    attributes = list(treated_matched_data.columns)
    attributes.remove("matched_element")
    for attr in attributes:
        untreated_matched_data[attr] = untreated_matched_data.apply(obtain_match_details, axis=1, all_data=df_data,
                                                                    attribute=attr)
    untreated_matched_data = untreated_matched_data.set_index('matched_element')

    ## create a df with all matched data
    all_mached_data = pd.concat([treated_matched_data, untreated_matched_data])

    with open(txt_pth, 'a') as f:
        f.write("\nExample treated matched data\n")
        df_temp = treated_matched_data.head(3)
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)

        f.write("\nExample untreated matched data\n")
        df_temp = untreated_matched_data.head(3)
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)

        f.write(f'\nShape untreated matched data: {untreated_matched_data.shape}\n')
        f.write(f'\nShape treated matched data: {treated_matched_data.shape}\n')

        f.write("\nAll  matched data value counts\n")
        df_temp = all_mached_data.treatment.value_counts()
        df_as_str = df_temp.to_string(header=True, index=True)
        f.write(df_as_str)
    f.close()

    plt_ind += 1
    # fig, ax = plt.subplots(2, 1)
    # fig.suptitle('Comparison of {} split by outcome and treatment status'.format('propensity_score_logit'))
    # sns.stripplot(data=df_data.groupby('treatment').apply(lambda x: x.sample(100)).reset_index(drop=True),
    #               y='outcome', x='propensity_score_logit', hue='treatment', orient='h',
    #               ax=ax[0]).set(title='Before matching', xlim=(-6, 4))
    # sns.stripplot(data=all_mached_data.groupby('treatment').apply(lambda x: x.sample(100)).reset_index(drop=True),
    #               y='outcome', x='propensity_score_logit', hue='treatment', ax=ax[1],
    #               orient='h').set(title='After matching', xlim=(-6, 4))
    # plt.subplots_adjust(hspace=0.3)
    # # plt.show()
    # plt.close()
    # fig.savefig(rf"{out_folder}\{plt_ind:02d}-jitter_plot.png")

    args = ["GROESSE", "Oeko", "energy", "farm_area", "Rinder", "Schweine", "Schafe", "ackerzahl", "dem", "slope",
            "soil_type", 'propensity_score_logit']

    plt_ind += 1
    # def plot(arg):
    #     fig, ax = plt.subplots(1, 2)
    #     fig.suptitle(f'Comparison of {arg} split by treatment status.')
    #     sns.kdeplot(data=df_data.groupby('treatment').apply(lambda x: x.sample(100)).reset_index(drop=True),
    #                 x=arg, hue='treatment', ax=ax[0]).set(title='Density before matching')
    #     sns.kdeplot(data=all_mached_data.groupby('treatment').apply(lambda x: x.sample(100)).reset_index(drop=True),
    #                 x=arg, hue='treatment', ax=ax[1]).set(title='Density after matching')
    #     # plt.show()
    #     plt.close()
    #     fig.savefig(rf"{out_folder}\{plt_ind:02d}-{arg}_comparison.png")
    #
    # for arg in args:
    #     plot(arg)

    data = []
    cols = attributes
    # cols = ['Age','SibSp','Parch','Fare','sex_female','sex_male','embarked_C','embarked_Q','embarked_S']
    for cl in cols:
        data.append([cl, 'before', cohenD(df_data, cl)])
        data.append([cl, 'after', cohenD(all_mached_data, cl)])

    res = pd.DataFrame(data, columns=['variable', 'matching', 'effect_size'])

    plt_ind += 1
    sn_plot = sns.barplot(data=res, y='variable', x='effect_size', hue='matching', orient='h')
    sn_plot.set(title='Standardised Mean differences across covariates before and after matching')
    sn_plot.figure.savefig(rf"{out_folder}\{plt_ind:02d}-standardised_mean_differences.png")

    overview = all_mached_data[['outcome', 'treatment']].groupby(by=['treatment']).aggregate(
        [np.mean, np.var, np.std, 'count'])

    treated_outcome = overview['outcome']['mean'][1]
    treated_counterfactual_outcome = overview['outcome']['mean'][0]

    att = treated_outcome - treated_counterfactual_outcome

    with open(txt_pth, 'a') as f:
        f.write("\nAll  matched data value counts\n")
        df_as_str = overview.to_string(header=True, index=True)
        f.write(df_as_str)
        f.write("\nThe Average Treatment Effect (ATT): {:.4f}\n".format(att))
    f.close()

    treated_outcome = treated_matched_data.outcome
    untreated_outcome = untreated_matched_data.outcome
    # stats_results = stats.ttest_ind(treated_outcome, untreated_outcome)

    tmp = pd.DataFrame(
        data={'treated_outcome': treated_outcome.values, 'untreated_outcome': untreated_outcome.values})

    return plt_ind



def propensity_score_matching():
    print("Propensity score matching for fields")
    plt.style.use('classic')
    sns.set(rc={'figure.figsize': (16, 10)}, font_scale=1.3)

    ## Define some settings
    year = 2020
    excl_attr = "medium"
    treatment_var_dict = {"small": 0, "big": 1}
    descr = f"_only_bb_farms_{list(treatment_var_dict.keys())[0]}_vs_{list(treatment_var_dict.keys())[1]}"

    print("\tRead and prepare data")
    df_fields = pd.read_csv(r"Q:\FORLand\Clemens\chapter02\data\tables\all_fields_with_values_2020.csv", dtype={"BTNR": str})
    if "c_factor" in list(df_fields.columns):
        df_fields["c_factor"] = df_fields["c_factor"] / 100
    df_fields.rename(columns={"ndvi-c-factor": "ndvi_c_factor"}, inplace=True)
    df_fields["farm_size_class"] = pd.cut(df_fields["farm_area"], bins=[0, 50, 100, 250, 500, 1000, 2500, 10000],
                                          labels=["<50", "<100", "<250", "<500", "<1000", "<2500", "<10000"])
    df_fields["field_size_class"] = pd.cut(df_fields["GROESSE"], bins=[0, 1, 2, 4, 8, 16, 32, 64, 250],
                                          labels=["<1", "<2", "<4", "<8", "<16", "<32", "<64", "<250"])

    print("No. fields before excluding non-BB farms:", len(df_fields))
    df_fields = df_fields.loc[df_fields["BTNR"].str.slice(0, 2) == '12'].copy()
    print("After:",  len(df_fields))

    df_red = df_fields[
        ["BTNR", "ID", "ID_KTYP", "GROESSE", "Oeko", "num_comp_c", "energy", "farm_area", "Rinder", "Schweine",
         "Schafe", "ackerzahl", "dem", "c_factor", "ndvi_c_factor", "slope", "soil_type", "farm_size_class",
         "field_size_class"]].copy()

    df_ktyp_areas = df_red.groupby(["BTNR", "ID_KTYP"]).agg(
        area=pd.NamedAgg("GROESSE", "sum")
    ).reset_index()
    df_ktyp_areas["share"] = 100 * df_ktyp_areas['area'] / df_ktyp_areas.groupby('BTNR')['area'].transform('sum')
    df_ktyp_areas = df_ktyp_areas.pivot(index='BTNR', columns='ID_KTYP', values='share')
    df_ktyp_areas.columns = [f"share_{int(col)}" for col in df_ktyp_areas.columns]
    df_ktyp_areas.fillna(0, inplace=True)
    df_ktyp_areas.reset_index(inplace=True)

    df_st_areas = df_red.groupby(["BTNR", "soil_type"]).agg(
        area=pd.NamedAgg("GROESSE", "sum")
    ).reset_index()
    df_st_areas["share"] = 100 * df_st_areas['area'] / df_st_areas.groupby('BTNR')['area'].transform('sum')
    df_st_areas = df_st_areas.pivot(index='BTNR', columns='soil_type', values='share')
    df_st_areas.columns = [f"share_{int(col)}" for col in df_st_areas.columns]
    df_st_areas.fillna(0, inplace=True)
    df_st_areas.reset_index(inplace=True)

    df_red.to_csv(fr"Q:\FORLand\Clemens\chapter02\data\tables\df_data.csv", sep=";", index=False)
    # df_red = df_red.loc[df_red["ID_KTYP"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30, 60])].copy()
    df_farms = df_red.groupby("BTNR").agg(
        num_comp_c=pd.NamedAgg("num_comp_c", "first"),
        energy=pd.NamedAgg("energy", "first"),
        farm_area=pd.NamedAgg("GROESSE", "sum"),
        Rinder=pd.NamedAgg("Rinder", "first"),
        Schweine=pd.NamedAgg("Schweine", "first"),
        Schafe=pd.NamedAgg("Schafe", "first"),
        ackerzahl=pd.NamedAgg("ackerzahl", "mean"),
        dem=pd.NamedAgg("dem", "mean"),
        c_factor=pd.NamedAgg("c_factor", "mean"),
        ndvi_c_factor=pd.NamedAgg("ndvi_c_factor", "mean"),
        slope=pd.NamedAgg("slope", "mean"),
        farm_size_class=pd.NamedAgg("farm_size_class", "first"),
        Oeko=pd.NamedAgg("Oeko", "sum")
    ).reset_index()

    df_farms.loc[df_farms["Oeko"] > 0, "Oeko"] = 1
    df_farms = pd.merge(df_farms, df_ktyp_areas, on="BTNR", how="left")
    df_farms = pd.merge(df_farms, df_st_areas, on="BTNR", how="left")
    df_farms["big_netw"] = 0
    df_farms.loc[df_farms["num_comp_c"] == "big", "big_netw"] = 1
    df_farms["medium_netw"] = 0
    df_farms.loc[df_farms["num_comp_c"] == "medium", "medium_netw"] = 1
    df_farms["single_farm"] = 0
    df_farms.loc[df_farms["num_comp_c"] == "small", "single_farm"] = 1
    df_farms.to_csv(fr"Q:\FORLand\Clemens\chapter02\data\tables\df_data_farms.csv", sep=";", index=False)

    for c_factor_col in ["c_factor", "ndvi_c_factor"]:
        print(f"######################################## {c_factor_col} ########################################")
        out_folder = fr"Q:\FORLand\Clemens\chapter02\data\results\{c_factor_col}_{year}{descr}"
        processing_lib.create_folder(out_folder)

        df_red = df_fields[
            ["BTNR", "ID", "ID_KTYP", "GROESSE", "Oeko", "num_comp_c", "energy", "farm_area", "Rinder", "Schweine",
             "Schafe", "ackerzahl", "dem", c_factor_col, "slope", "soil_type", "farm_size_class",
             "field_size_class"]].copy()  # ,"focus_farms"
        df_red.dropna(inplace=True)
        df = df_red.copy()
        df.rename(columns={"num_comp_c": "treatment"}, inplace=True)
        df = df.loc[df["ID_KTYP"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30, 60])].copy()

        # plt_ind = 0
        # plt_ind = data_exploration(
        #     df=df,
        #     out_folder=out_folder,
        #     c_factor_col=c_factor_col,
        #     plt_ind=plt_ind,
        #     year=year)
        #
        # plt_ind = prop_match_wrapper(
        #     df=df,
        #     c_factor_col=c_factor_col,
        #     out_folder=out_folder,
        #     excl_attr=excl_attr,
        #     treatment_var_dict=treatment_var_dict,
        #     plt_ind=plt_ind,
        #     year=year)


        ##################### Logistic Regression of c-factor~farm and soil characteristics #####################


        # y = df_data.ndvi_c_factor
        # X = df_data.loc[:, df_data.columns != c_factor_col]
        #
        # ## Convert categorical variables into dummy variables
        # X_encoded = pd.get_dummies(
        #     data=X,
        #     columns=["Oeko", "energy", "soil_type", "treatment"],
        #     prefix={'Oeko': 'eco', "energy": "energy", "soil_type": "soil", "treatment": "big_netw"}, drop_first=False)
        #
        # from sklearn.linear_model import TweedieRegressor
        #
        # ## Design pipeline to build the treatment estimator. It standardizes the data and applies a logistic classifier
        # pipe = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('logistic_classifier', TweedieRegressor())
        # ])
        #
        # pipe.fit(X_encoded, y)
        # predictions = pipe.predict_proba(X_encoded)

        print("Done")

def main():
    stime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)

    ## Resample and reproject raster data
    ## R-factor
    # for year in range(2017, 2021):
    #     processing_lib.gdal_warp_wrapper(
    #         input_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\R_FAKTOR_2001_2021_RADKLIM_v2017_002_TIFF_GK3_DE10km\R_Faktoren_DE_D60_V2017_002_SW264_2001_2021_RUN01_{year}_GK3_DE10kmBuffer.tif",
    #         ref_ras_pth=r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #         output_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\R_FAKTOR_2001_2021_RADKLIM_v2017_002_TIFF_GK3_DE10km\R_Faktor_{year}_3035_DE10kmBuffer_10m.tif"
    #     )
    #
    # ## K-factor
    # processing_lib.gdal_warp_wrapper(
    #     input_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\Bodenart\bodenart_kfaktor_klassifiziert_25832.tif",
    #     ref_ras_pth=r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #     output_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\Bodenart\bodenart_kfaktor_3035_10m_temp.tif"
    # )
    #
    # ## LS-factor
    # processing_lib.gdal_warp_wrapper(
    #     input_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\LS_factor.tif",
    #     ref_ras_pth=r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #     output_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\LS_factor_10m_3035.tif"
    # )

    ## Ackerzahl
    # processing_lib.gdal_warp_wrapper(
    #     input_ras_pth=fr"Q:\FORLand\Daten\raster\ALKIS_bodenschaetzung\raster_5m_ackerzahl\ackerzahl_bodenschaetzung_BB_5m.tif",
    #     ref_ras_pth=r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #     output_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\ackerzahl_bodenschaetzung_10m_3035.tif",
    #     input_epsg="25832"
    # )

    # processing_lib.aggregate_raster_values_by_raster_mask(
    #     input_ras_pth=r"data\raster\ackerzahl_bodenschaetzung_10m_3035.tif",
    #     mask_ras_pth=r"data\raster\field_ids-2020.tif",
    #     output_pth=r"data\tables\extract_values\IACS_BB_2020_ackerzahl.csv",
    #     column_names=["ID", "ackerzahl"])

    # processing_lib.aggregate_raster_values_by_raster_mask(
    #     input_ras_pth=r"data\raster\DEM_GER_10m_clipped_3035.tif",
    #     mask_ras_pth=r"data\raster\field_ids-2020.tif",
    #     output_pth=r"data\tables\extract_values\IACS_BB_2020_dem.csv",
    #     column_names=["ID", "dem"])
    #
    # processing_lib.aggregate_raster_values_by_raster_mask(
    #     input_ras_pth=r"data\raster\Slope_10m_3035_grad.tif",
    #     mask_ras_pth=r"data\raster\field_ids-2020.tif",
    #     output_pth=r"data\tables\extract_values\IACS_BB_2020_slope.csv",
    #     column_names=["ID", "slope"])

    # processing_lib.aggregate_raster_values_by_raster_mask(
    #     input_ras_pth=r"data\raster\Bodenart\bodenart_bodenschaetzung_BB_10m.tif",
    #     mask_ras_pth=r"data\raster\field_ids-2020.tif",
    #     output_pth=r"data\tables\extract_values\IACS_BB_2020_soil_type.csv",
    #     column_names=["ID", "soil_type"],
    #     aggfunc="mode"
    # )

    # processing_lib.aggregate_raster_values_by_raster_mask(
    #     input_ras_pth=r"data\raster\c_factor\c_factor_2020_3035_10m.tif",
    #     mask_ras_pth=r"data\raster\field_ids-2020.tif",
    #     output_pth=r"data\tables\extract_values\IACS_BB_2020_c_factor.csv",
    #     column_names=["ID", "c_factor"],
    #     aggfunc="mean"
    # )

    # processing_lib.aggregate_raster_values_by_raster_mask(
    #     input_ras_pth=r"data\raster\A_USLE\A_2020-R_Radklim-K_Bodensch-LS_Copern-C_NDVI-3035_10m.tif",
    #     mask_ras_pth=r"data\raster\field_ids-2020.tif",
    #     output_pth=r"data\tables\extract_values\IACS_BB_2020_A_2020-R_Radklim-K_Bodensch-LS_Copern-C_NDVI.csv",
    #     column_names=["ID", "A"],
    #     aggfunc="mean"
    # )

    ## Merge dataframe
    # merge_dataframes()

    ## Propensity score matching
    propensity_score_matching()


    etime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)
    print("end: " + etime)


if __name__ == '__main__':
    main()