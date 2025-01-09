import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import regex as re
from copy import deepcopy
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import seaborn as sns
import sys
import pipeline_utilities as pu
import clinical_trial_classes as ctc
import data_loading_functions as dlf
import matplotlib.pyplot as plt
import random
from lifelines.plotting import add_at_risk_counts
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import miceforest as mf

def pairwise_log_rank(clinical_trials, km_analyses, med_list=None):
    """Perform pairwise log-rank tests between all medications (and cohorts) in clinical_trials"""
    if med_list == None:
        med_list = list(clinical_trials.keys())

    #construct pairwise comparisons
    pairs = list(itertools.combinations([med for med in list(clinical_trials.keys()) if med in med_list], 2))
    
    #for each pair, perform logrank test and print results
    all_results = {}
    for pair in pairs:
        all_results[pair] = logrank_test(
            durations_A=km_analyses[pair[0]].durations, 
            durations_B=km_analyses[pair[1]].durations, 
            event_observed_A=km_analyses[pair[0]].event_observed,
            event_observed_B=km_analyses[pair[1]].event_observed,
            weights_A=km_analyses[pair[0]].weights,
            weights_B=km_analyses[pair[1]].weights, 
            )
        print(f"Comparison between {pair}: p = {all_results[pair].p_value}, Adj. p = {all_results[pair].p_value * len(pairs)}")

    return all_results

def enforce_positivity_on_subjects(trial_confounders, clinical_trials, trial_tables, enforce_IPTW, enforce_IPCW):
    """Make sure all subjects have positivity in propensity scores by removing outliers"""
    for med in trial_confounders:
        if enforce_IPTW:
            #get propensity scores of in and out of trial subjects
            in_trial_scores = trial_confounders[med].loc[trial_confounders[med].in_trial == 1].propensity_IPTW
            out_trial_scores = trial_confounders[med].loc[trial_confounders[med].in_trial == 0].propensity_IPTW
    
            #calculate the range of the propensity scores
            in_trial_range = [np.min(in_trial_scores), np.max(in_trial_scores)]
            out_trial_range = [np.min(out_trial_scores), np.max(out_trial_scores)]
    
            #enforce a range between the maximum min, and minimum max
            enforced_range_IPTW = [np.max([in_trial_range[0], out_trial_range[0]]), np.min([in_trial_range[1], out_trial_range[1]])]
            trial_confounders[med] = trial_confounders[med].loc[(trial_confounders[med].propensity_IPTW >= enforced_range_IPTW[0]) & (trial_confounders[med].propensity_IPTW <= enforced_range_IPTW[1])]
            clinical_trials[med].cohort = [subject for subject in clinical_trials[med].cohort if (subject.propensity_score['IPTW'] >= enforced_range_IPTW[0] and subject.propensity_score['IPTW'] <= enforced_range_IPTW[1])]
            trial_tables[med] = trial_tables[med].loc[trial_tables[med].MRN.isin([subject.patient.pat_id for subject in clinical_trials[med].cohort])]

        if enforce_IPCW:
            #get propensity scores of censored and uncensored in-trial subjects
            censored_scores = trial_confounders[med].loc[(trial_confounders[med].in_trial == 1) & (trial_confounders[med].is_censored == 1)].propensity_IPCW
            uncensored_scores = trial_confounders[med].loc[(trial_confounders[med].in_trial == 1) & (trial_confounders[med].is_censored == 0)].propensity_IPCW

            #calculate the range of the propensity scores
            censored_range = [np.min(censored_scores), np.max(censored_scores)]
            uncensored_range = [np.min(uncensored_scores), np.max(uncensored_scores)]

            #enforce a range between the maximum min, and minimum max
            enforced_range_IPCW = [np.max([censored_range[0], uncensored_range[0]]), np.min([censored_range[1], uncensored_range[1]])]
            trial_confounders[med].loc[trial_confounders[med].in_trial == 1] = trial_confounders[med].loc[(trial_confounders[med].in_trial == 1) & (trial_confounders[med].propensity_IPCW >= enforced_range_IPCW[0]) & (trial_confounders[med].propensity_IPCW <= enforced_range_IPCW[1])]
            trial_confounders[med] = trial_confounders[med].dropna(how='all')
                                                                   
            clinical_trials[med].cohort = [subject for subject in clinical_trials[med].cohort if (subject.propensity_score['IPCW'] >= enforced_range_IPCW[0] and subject.propensity_score['IPCW'] <= enforced_range_IPCW[1])]
            
            trial_tables[med] = trial_tables[med].loc[trial_tables[med].MRN.isin([subject.patient.pat_id for subject in clinical_trials[med].cohort])]
    
    return trial_confounders, clinical_trials, trial_tables

def calculate_propensity_scores(clinical_trials, use_IPTW, use_IPCW, trial_tables, max_duration_days):
    """Calculate propensity scores using covariates. IPTW uses Gradient Boosted Models, IPCW uses Logistic Regression"""
    #for each trial, gather the trial cohort, and the non-trial cohort (all other patients) confounders into a table
    gb_models = {}
    all_trial_confounders = {}
    for trial_med in clinical_trials:
        #get the confounders of patients in the trial
        trial_confounders = []
        for subject in clinical_trials[trial_med].cohort:
            #add their trial information
            trial_confounders.append(deepcopy(subject.confounders))
            trial_confounders[-1]['in_trial'] = 1
            trial_confounders[-1]['MRN'] = subject.patient.pat_id

            #add their censoring information
            #patient is censored if in trial_tables, they do not have the event before max_duration_days and they don't have any post_trial_visits
            patient_trial_table = trial_tables[trial_med].loc[trial_tables[trial_med].MRN == subject.patient.pat_id]
            if len(patient_trial_table) > 1:
                raise
            trial_confounders[-1]['is_censored'] = (bool(patient_trial_table.event.iloc[0]) == False) and (patient_trial_table.duration.iloc[0] < max_duration_days) # and (len(subject.post_trial_visits) < 1)
    
        #get the non-trial confounders
        nontrial_confounders = []
        for nontrial_med in clinical_trials:
            if nontrial_med == trial_med:
                continue
            for subject in clinical_trials[nontrial_med].cohort:
                nontrial_confounders.append(deepcopy(subject.confounders))
                nontrial_confounders[-1]['in_trial'] = 0
                nontrial_confounders[-1]['MRN'] = subject.patient.pat_id

                #patient is censored if in trial_tables, they do not have the event before max_duration_days and they don't have any post_trial_visits
                patient_nontrial_table = trial_tables[nontrial_med].loc[trial_tables[nontrial_med].MRN == subject.patient.pat_id]
                if len(patient_nontrial_table) > 1:
                    raise
                nontrial_confounders[-1]['is_censored'] = (bool(patient_nontrial_table.event.iloc[0]) == False) and (patient_nontrial_table.duration.iloc[0] < max_duration_days) # and (len(subject.post_trial_visits) < 1)
                # nontrial_confounders[-1]['is_censored'] = np.nan

        #convert to DF
        trial_confounders = pd.DataFrame(trial_confounders)
        nontrial_confounders = pd.DataFrame(nontrial_confounders)

        #concatenate and then drop missing values
        all_confounders = pd.concat([trial_confounders, nontrial_confounders], ignore_index=True)
    
        #create datasets for propensity scores for IPTW
        x_iptw = all_confounders.drop(['in_trial', 'MRN', 'is_censored'], axis=1)
        y_iptw = all_confounders.in_trial

        #create datasets for propensity scores for IPCW 
        x_ipcw = trial_confounders.drop(['in_trial', 'MRN', 'is_censored'], axis=1)
        y_ipcw = trial_confounders.is_censored

        gb_models[trial_med] = {}
        
        #calculate propensity scores for IPTW
        if use_IPTW:
            gbm_IPTW = GradientBoostingClassifier().fit(x_iptw, y_iptw)
            all_confounders['propensity_IPTW'] = gbm_IPTW.predict_proba(x_iptw)[:,1]
            #collect the gbm_IPTW
            gb_models[trial_med]['IPTW'] = gbm_IPTW

        #calculate propensity scores for IPCW
        if use_IPCW:
            gbm_IPCW = LogisticRegression().fit(x_ipcw, y_ipcw)
            all_confounders['propensity_IPCW'] = gbm_IPCW.predict_proba(x_iptw)[:,1]            
            #collect the gbm_IPCW
            gb_models[trial_med]['IPCW'] = gbm_IPCW

        #calculate final propensity scores
        if use_IPTW and use_IPCW:
            all_confounders['propensity'] = all_confounders.apply(lambda x: x.propensity_IPTW * x.propensity_IPCW, axis=1)
        elif use_IPTW:
            all_confounders['propensity'] = all_confounders.propensity_IPTW
        elif use_IPCW:
            all_confounders['propensity'] = all_confounders.propensity_IPCW

        #add propensity scores to subjects
        for i in range(len(clinical_trials[trial_med].cohort)):
            #double check the MRNs line up
            if clinical_trials[trial_med].cohort[i].patient.pat_id != all_confounders.loc[i].MRN:
                raise
            clinical_trials[trial_med].cohort[i].propensity_score['overall'] = all_confounders.loc[i].propensity
            if use_IPTW:
                clinical_trials[trial_med].cohort[i].propensity_score['IPTW'] = all_confounders.loc[i].propensity_IPTW
            if use_IPCW:
                clinical_trials[trial_med].cohort[i].propensity_score['IPCW'] = all_confounders.loc[i].propensity_IPCW

        all_trial_confounders[trial_med] = all_confounders
        
    return gb_models, all_trial_confounders

def get_confounder_importances(gb_models, clinical_trials, propensity_set=None):
    """Get the weights of each covariate in the gradient boosted model to see their impact on treatment propensity"""
    proto_importance = []
    for med in gb_models:
        if propensity_set is None:
            importances = gb_models[med].feature_importances_
            features = list(clinical_trials[med].cohort[0].confounders.keys())
            proto_importance.append({features[i]:importances[i] for i in range(len(importances))})  
        else:
            if isinstance(gb_models[med][propensity_set], LogisticRegression):
                importances = gb_models[med][propensity_set].coef_.ravel()
            else:
                importances = gb_models[med][propensity_set].feature_importances_
            features = list(clinical_trials[med].cohort[0].confounders.keys())
            proto_importance.append({features[i]:importances[i] for i in range(len(importances))})
    return pd.DataFrame(proto_importance) 

def plot_iptw_positivity(trial_confounders, main_save_dir=None):
    """Plots the propensity score distributions for IPTW weighting for each ASM in the trial"""
    for med in trial_confounders:
        fig, axs = plt.subplots(1, 2, figsize=(10,3))
        in_trial_scores = trial_confounders[med].loc[trial_confounders[med].in_trial == 1]['propensity_IPTW']
        out_trial_scores = trial_confounders[med].loc[trial_confounders[med].in_trial == 0]['propensity_IPTW']
        axs[0].hist(in_trial_scores, bins=np.linspace(0, 1, 26), alpha = 0.33, density=True)
        axs[0].hist(out_trial_scores, bins=np.linspace(0, 1, 26), alpha = 0.33, density=True)
        axs[0].legend(["In trial", "Out of trial"])
        axs[0].set_xlabel("Propensity Score")
        
        axs[1].boxplot([in_trial_scores, out_trial_scores])
        axs[1].set_xticklabels(['In trial', 'Out of Trial'])
        axs[1].set_ylabel("Propensity Score")
        fig.suptitle(f"{med} IPTW")
        if main_save_dir is not None:
            plt.savefig(f"{main_save_dir}/{med}_IPTW_Positivity.png", dpi=600, bbox_inches='tight')
            plt.savefig(f"{main_save_dir}/{med}_IPTW_Positivity.pdf", dpi=600, bbox_inches='tight')
        plt.show()

def plot_ipcw_positivity(trial_confounders, main_save_dir=None):
    """Plots the propensity score distributions for PICW weighting for each ASM in the trial"""
    for med in trial_confounders:
        fig, axs = plt.subplots(1, 2, figsize=(10,3))
        is_censored_scores = trial_confounders[med].loc[(trial_confounders[med].in_trial == 1) & (trial_confounders[med].is_censored == 1)]['propensity_IPCW']
        is_uncensored_scores = trial_confounders[med].loc[(trial_confounders[med].in_trial == 1) & (trial_confounders[med].is_censored == 0)]['propensity_IPCW']
        axs[0].hist(is_censored_scores, bins=np.linspace(0, 1, 26), alpha = 0.33, density=True)
        axs[0].hist(is_uncensored_scores, bins=np.linspace(0, 1, 26), alpha = 0.33, density=True)
        axs[0].legend(["Is Censored", "Is not Censored"])
        axs[0].set_xlabel("Propensity Score")
        
        axs[1].boxplot([is_censored_scores, is_uncensored_scores])
        axs[1].set_xticklabels(["Is Censored", "Is not Censored"])
        axs[1].set_ylabel("Propensity Score")
        fig.suptitle(f"{med} IPCW")
        if main_save_dir is not None:
            plt.savefig(f"{main_save_dir}/{med}_IPCW_Positivity.png", dpi=600, bbox_inches='tight')
            plt.savefig(f"{main_save_dir}/{med}_IPCW_Positivity.pdf", dpi=600, bbox_inches='tight')
        plt.show()

def smd(mean_1, mean_2, var_1, var_2):
    """Formula for standard mean difference"""
    return np.abs(mean_1 - mean_2) / np.sqrt((var_1 + var_2)/2)

def smd_series(mean_1, mean_2, var_1, var_2):
    """Calculates standard mean difference, but for a pandas series"""
    return pd.Series({confound:smd(mean_1[confound], mean_2[confound], var_1[confound], var_2[confound]) for confound in var_1.index})

def weighted_var(weights, values, weighted_mean):
    """Calculates weighted variance using a weighted mean"""
    numerator = np.sum(weights*(values-weighted_mean))
    denominator = np.sum(weights)
    return numerator/denominator

def weighted_var_series(weighted_confounders):
    """Calculates weighted variance for a pandas series"""
    weighted_mean = weighted_confounders.sum()/weighted_confounders.weight.sum()
    return pd.Series({confound:weighted_var(weighted_confounders.weight, weighted_confounders[confound], weighted_mean[confound]) for confound in weighted_confounders.columns})        

def plot_pre_post_weighting_bias_balance_iptw(trial_confounders, void=True, main_save_dir=None):
    """Plots the bias balance before and after IPTW weighting"""
    unweighted_balances = {}
    weighted_balances = {}

    #for each med, calculate the weighted and unweighted balances
    for med in trial_confounders:
        #get the weights
        in_trial_weights = truncate_IP_weights(1/trial_confounders[med].loc[trial_confounders[med].in_trial==1]['propensity_IPTW'])
        out_trial_weights = truncate_IP_weights(1/(1 - trial_confounders[med].loc[trial_confounders[med].in_trial==0]['propensity_IPTW']))
        weights = in_trial_weights + out_trial_weights


        #get the unweighted and weighted values of the covariates
        drop_columns = ['MRN', 'propensity', 'is_censored']
        if 'propensity_IPTW' in trial_confounders[med].columns:
            drop_columns.append('propensity_IPTW')
        if 'propensity_IPCW' in trial_confounders[med].columns:
            drop_columns.append('propensity_IPCW')
        unweighted_confounders = trial_confounders[med].drop(drop_columns, axis=1)
        weighted_confounders = deepcopy(unweighted_confounders).mul(weights, axis=0)
        weighted_confounders['weight'] = weights
    
        #calculate means
        unweighted_in_trial_mean = unweighted_confounders.loc[unweighted_confounders.in_trial > 0].mean()
        unweighted_out_trial_mean = unweighted_confounders.loc[unweighted_confounders.in_trial == 0].mean()
        weighted_in_trial_mean = weighted_confounders.loc[weighted_confounders.in_trial > 0].sum()/weighted_confounders.loc[weighted_confounders.in_trial > 0].weight.sum()
        weighted_out_trial_mean = weighted_confounders.loc[weighted_confounders.in_trial == 0].sum()/weighted_confounders.loc[weighted_confounders.in_trial == 0].weight.sum()
    
        #calculate variances of pooled sample
        unweighted_in_trial_var = unweighted_confounders.loc[unweighted_confounders.in_trial > 0].var()
        unweighted_out_trial_var = unweighted_confounders.loc[unweighted_confounders.in_trial == 0].var()
        weighted_in_trial_var = weighted_var_series(weighted_confounders.loc[weighted_confounders.in_trial > 0])
        weighted_out_trial_var = weighted_var_series(weighted_confounders.loc[weighted_confounders.in_trial == 0])

        #calculate standardized mean differences
        unweighted_balances[med] = smd_series(unweighted_in_trial_mean, unweighted_out_trial_mean, unweighted_in_trial_var, unweighted_out_trial_var).drop(['in_trial'])
        weighted_balances[med] = smd_series(weighted_in_trial_mean, weighted_out_trial_mean, weighted_in_trial_var, weighted_out_trial_var).drop(['in_trial', 'weight'])
    
    #plot balances
    for med in unweighted_balances:
        plt.figure()
        
        plt.ylabel("Standardized Bias")
        plt.scatter(range(len(unweighted_balances[med])), unweighted_balances[med], alpha=0.5)
        plt.scatter(range(len(weighted_balances[med])), weighted_balances[med], alpha=0.5)
        plt.legend(["Unweighted", "Weighted"])
        plt.hlines(0.2, 0, len(unweighted_balances[med].index), colors='r', linestyles='dashed')
        plt.hlines(0.1, 0, len(unweighted_balances[med].index), colors='g', linestyles='dashed')
        plt.xticks(range(len(unweighted_balances[med])), labels=unweighted_balances[med].index, rotation=90)
        plt.title(f"Bias Balance {med}, IPTW")
        plt.ylim([0, 1])
        if main_save_dir is not None:
            plt.savefig(f"{main_save_dir}/{med}_IPTW_balance.png", dpi=600, bbox_inches='tight')
            plt.savefig(f"{main_save_dir}/{med}_IPTW_balance.pdf", dpi=600, bbox_inches='tight')
        plt.show()

    if not void:
        return unweighted_balances, weighted_balances

def plot_pre_post_weighting_bias_balance_ipcw(trial_confounders, void=True, main_save_dir=None):   
    """Plots the bias balance before and after IPCW weighting"""
    unweighted_balances = {}
    weighted_balances = {}

    #for each med, calculate the weighted and unweighted balances
    for med in trial_confounders:
        is_censored_weights = truncate_IP_weights(1/trial_confounders[med].loc[(trial_confounders[med].in_trial == 1) & (trial_confounders[med].is_censored == 1)]['propensity_IPCW'])
        not_censored_weights = truncate_IP_weights(1/(1 - trial_confounders[med].loc[(trial_confounders[med].in_trial == 1) & (trial_confounders[med].is_censored == 0)]['propensity_IPCW']))
        weights = is_censored_weights + not_censored_weights

        #get the unweighted and weighted values of the covariates
        drop_columns = ['MRN', 'propensity', 'in_trial']
        if 'propensity_IPTW' in trial_confounders[med].columns:
            drop_columns.append('propensity_IPTW')
        if 'propensity_IPCW' in trial_confounders[med].columns:
            drop_columns.append('propensity_IPCW')
        unweighted_confounders = trial_confounders[med].loc[trial_confounders[med].in_trial == 1].drop(drop_columns, axis=1)
        weighted_confounders = deepcopy(unweighted_confounders).mul(weights, axis=0)
        weighted_confounders['weight'] = weights
    
        #calculate means
        unweighted_is_censored_mean = unweighted_confounders.loc[unweighted_confounders.is_censored > 0].mean()
        unweighted_not_censored_mean = unweighted_confounders.loc[unweighted_confounders.is_censored == 0].mean()
        weighted_is_censored_mean = weighted_confounders.loc[weighted_confounders.is_censored > 0].sum()/weighted_confounders.loc[weighted_confounders.is_censored > 0].weight.sum()
        weighted_not_censored_mean = weighted_confounders.loc[weighted_confounders.is_censored == 0].sum()/weighted_confounders.loc[weighted_confounders.is_censored == 0].weight.sum()
    
        #calculate variances of pooled sample
        unweighted_is_censored_var = unweighted_confounders.loc[unweighted_confounders.is_censored > 0].var()
        unweighted_not_censored_var = unweighted_confounders.loc[unweighted_confounders.is_censored == 0].var()
        weighted_is_censored_var = weighted_var_series(weighted_confounders.loc[weighted_confounders.is_censored > 0])
        weighted_not_censored_var = weighted_var_series(weighted_confounders.loc[weighted_confounders.is_censored == 0])

        #calculate standardized mean differences
        unweighted_balances[med] = smd_series(unweighted_is_censored_mean, unweighted_not_censored_mean, unweighted_is_censored_var, unweighted_not_censored_var).drop(['is_censored'])
        weighted_balances[med] = smd_series(weighted_is_censored_mean, weighted_not_censored_mean, weighted_is_censored_var, weighted_not_censored_var).drop(['is_censored', 'weight'])
    
    #plot balances
    for med in unweighted_balances:
        plt.figure()
        
        plt.ylabel("Standardized Bias")
        plt.scatter(range(len(unweighted_balances[med])), unweighted_balances[med], alpha=0.5)
        plt.scatter(range(len(weighted_balances[med])), weighted_balances[med], alpha=0.5)
        plt.legend(["Unweighted", "Weighted"])
        plt.hlines(0.2, 0, len(unweighted_balances[med].index), colors='r', linestyles='dashed')
        plt.hlines(0.1, 0, len(unweighted_balances[med].index), colors='g', linestyles='dashed')
        plt.xticks(range(len(unweighted_balances[med])), labels=unweighted_balances[med].index, rotation=90)
        plt.title(f"Bias Balance {med}, IPCW")
        plt.ylim([0, 1])
        if main_save_dir is not None:
            plt.savefig(f"{main_save_dir}/{med}_IPCW_balance.png", dpi=600, bbox_inches='tight')
            plt.savefig(f"{main_save_dir}/{med}_IPCW_balance.pdf", dpi=600, bbox_inches='tight')
        plt.show()

    if not void:
        return unweighted_balances, weighted_balances

def truncate_IP_weights(proto_weights, lower_p=1, upper_p=99):
    """truncates propensity score weights between lower_p and upper_p percentiles"""
    p1 = np.percentile(proto_weights,1)
    p99 = np.percentile(proto_weights,99)
    return [np.max([p1, np.min([weight, p99])]) for weight in proto_weights]

def construct_survival_table_retention(clinical_trial, max_duration_days=None):
    """
    We want to create a KM table, where:
        the rows are patients
        column 0 is how long the patient is in the curve
        column 1 is if the patient had the event (stopped taking the new ASM)
    Patients who do not have the event but stay until the end of their prescription are censored
        the only outcome is if they stop the drug, regardless of the other drugs
    Patients who have the event "die" at time column 0.
    """
    #preallocate the table
    trial_table = {'duration':np.full(len(clinical_trial.cohort), np.nan), 'event':np.full(len(clinical_trial.cohort), np.nan), 'MRN':np.full(len(clinical_trial.cohort), "", dtype=object)}

    #for each subject
    for i in range(len(clinical_trial.cohort)):
        subject = clinical_trial.cohort[i]

        #a patient stops taking the drug if they have visits after stopping the drug
        #we do not care if they start a new drug or not
        #get the time the patient was in the trial
        trial_table['MRN'][i] = subject.patient.pat_id
        trial_table['duration'][i] = subject.time_in_trial.days
        #if there are post-trial visits, then the patient stopped taking the drug and therefore had the event
        trial_table['event'][i] = (len(subject.post_trial_visits) > 0)
        
        #censor patients who have data beyond max_duration days
        if max_duration_days != None:
            if trial_table['duration'][i] > max_duration_days:
                trial_table['duration'][i] = max_duration_days
                trial_table['event'][i] = False
    return pd.DataFrame(trial_table)

def skmice_impute(confounder_df, metadata_cols):
    """MICE imputation, but with SKlearn's implementation"""
    #we need to deepcopy everything first
    confounder_df = deepcopy(confounder_df)
    confounder_imputer = IterativeImputer(initial_strategy='most_frequent')
    only_confounders = confounder_df.drop(metadata_cols, axis='columns')
    only_metadata = confounder_df[metadata_cols]
    imputed_confounders = confounder_imputer.fit_transform(only_confounders)
    imputed_confounders = pd.DataFrame(imputed_confounders, columns=only_confounders.columns)
    return pd.concat([imputed_confounders, only_metadata], axis='columns')

def mice_impute(confounder_df, metadata_cols):
    """Mice imputation, but using Mice Forest"""
    kds = mf.ImputationKernel(confounder_df.drop(metadata_cols, axis='columns'), save_all_iterations=False)
    kds.mice(5, verbose=False, num_threads=7)
    imputed_confounders = kds.complete_data()
    return pd.concat([confounder_df[metadata_cols], imputed_confounders], axis='columns')

def empirical_impute(confounder_df):
    """Replace missing values with False"""
    #we need to deepcopy everything first
    confounder_df = deepcopy(confounder_df)
    confounder_df.status_epilepticus.fillna(False, inplace=True)
    confounder_df.febrile_history.fillna(False, inplace=True)
    confounder_df.intellectual_disability.fillna(False, inplace=True)
    return confounder_df

def impute_confounders(confounder_df, metadata_cols, imputation_strategy):
    """Impute missing confounders with the specified imputation_strategy"""
    if imputation_strategy is None:
        pass
    elif imputation_strategy.upper() == "MICE":
        confounder_df = mice_impute(confounder_df, metadata_cols)
    elif imputation_strategy.upper() == "SK_MICE":
        confounder_df = skmice_impute(confounder_df, metadata_cols)
    elif imputation_strategy.lower() == "empirical":
        confounder_df = empirical_impute(confounder_df)
    else:
        raise ValueError("Error - unknown Imputation Method. Choose from 'Empirical', 'SK_MICE', or 'MICE'")
    return confounder_df.dropna()

def calculate_propensity_weights(clinical_trial):
    """Calculate weights from propensity scores, and truncate to be between 1st and 99th percentiles"""
    weights = pd.DataFrame([{'MRN':subject.patient.pat_id, 'propensity':subject.propensity_score['overall']} for subject in clinical_trial.cohort])
    weights['weight'] = truncate_IP_weights(1/weights.propensity)
    return weights.drop(['propensity'], axis=1)

def retention_survival(all_agg_pats, med_list, minimum_prior_asms, maximum_prior_asms, min_cohort_size, 
                       medication_start_regex, medication_end_regex, brand_to_generic, max_duration_days=None, shadow_propensities=False,
                       use_IPTW=False, use_IPCW=False, confounder_imputation='MICE', confounder_table=None, metadata_cols=None, enforce_positivity=True):
    """Conduct the retention clinical trial with the specified parameters"""

    #initialize clinical trial objects and impute missing confounders
    clinical_trials = {}
    for med in med_list:        
        new_trial = ctc.ClinicalTrial(med, 
                                      outcome_measure='retention', 
                                      minimum_prior_asms=minimum_prior_asms, 
                                      maximum_prior_asms=maximum_prior_asms)
        enroll_sub_stat = new_trial.enroll_subjects(all_agg_pats, medication_start_regex, medication_end_regex, brand_to_generic, debug=False)
        if use_IPTW or use_IPCW or shadow_propensities:
            imputed_confounder_table = impute_confounders(confounder_table, metadata_cols, confounder_imputation)
            new_trial.cohort = [subject for subject in new_trial.cohort if subject.determine_confounders(imputed_confounder_table, metadata_cols) == 1]
        if len(new_trial.cohort) >= min_cohort_size:
            clinical_trials[med] = new_trial

    #create survival tables
    trial_tables = {}
    for med in clinical_trials:
        trial_tables[med] = construct_survival_table_retention(clinical_trials[med], max_duration_days)

    #calculate propensity scores
    if use_IPTW or use_IPCW:
        gb_models, trial_confounders = calculate_propensity_scores(clinical_trials, use_IPTW, use_IPCW, trial_tables, max_duration_days)
        if enforce_positivity:
            trial_confounders, clinical_trials, trial_tables = enforce_positivity_on_subjects(trial_confounders, clinical_trials, trial_tables, enforce_IPTW=use_IPTW, enforce_IPCW=use_IPCW)
    elif shadow_propensities:
        gb_models, trial_confounders = calculate_propensity_scores(clinical_trials, True, True, trial_tables, max_duration_days)
        if enforce_positivity:
            trial_confounders, clinical_trials, trial_tables = enforce_positivity_on_subjects(trial_confounders, clinical_trials, trial_tables, enforce_IPTW=True, enforce_IPCW=True)
    
    #for each trial, create and fit the KM models
    km_analyses = {}
    for med in clinical_trials:
        km_model = KaplanMeierFitter()
        if use_IPTW or use_IPCW:
            #calculate weights, and truncate to be between 1st and 99th percentiles
            weights = calculate_propensity_weights(clinical_trials[med])
            trial_tables[med] = trial_tables[med].merge(weights, on='MRN', how='inner')
            km_model.fit(trial_tables[med]['duration'], trial_tables[med]['event'], label=med, weights=trial_tables[med]['weight'])
        else:
            km_model.fit(trial_tables[med]['duration'], trial_tables[med]['event'], label=med)
        km_analyses[med] = km_model

    if use_IPTW or use_IPCW or shadow_propensities:
        return clinical_trials, trial_tables, gb_models, trial_confounders, km_analyses
    else:
        return clinical_trials, trial_tables, km_analyses

def adjust_hasSz_by_elo(vis, trial_start_time):
    """if the algorithm thought the patient was having seizures, and a date of last seizure was recorded,
    check when the last seizure was. If it was before the trial start time, then override the hasSz prediction"""
    if (vis.hasSz == 1 or vis.hasSz == 2) and not np.any(pd.isnull(vis.elo)):
        if vis.elo < trial_start_time:
            return 0
    return vis.hasSz

def construct_survival_table_szFreedom(clinical_trial, idk_behavior, max_duration_days=None):
    """
    We want to create a KM table, where:
        the rows are patients
        column 0 is how long the patient is in the curve
        column 1 is if the patient had the event (breakthrough seizure)
    patients who do not have the event but stay the max length "survive"
    patients who do not have the event but stay < max length "drop out"
    patients who have the event "die" at time column 0.
    """
    #preallocate the table
    trial_table = {'duration':np.full(len(clinical_trial.cohort), np.nan), 'event':np.full(len(clinical_trial.cohort), np.nan), 'MRN':np.full(len(clinical_trial.cohort), "", dtype=object)}

    #a container to store patients who have only visits with idk classifications
    pat_idx_empty_by_idk = []
    
    #for each subject
    for i in range(len(clinical_trial.cohort)):
        subject = clinical_trial.cohort[i]
        
        #when did the patient join and leave the trial?
        trial_start_time = subject.patient.medications[clinical_trial.name].start_date
        trial_end_time = trial_start_time + subject.time_in_trial
        
        #for each of their trial visits, track when they are seizurefree
        subject_status = {'date':[], 'hasSz':[]}
        for vis in subject.trial_visits:
            subject_status['date'].append(vis.visit_date)
            subject_status['hasSz'].append(adjust_hasSz_by_elo(vis, trial_start_time))
        
        #if there are post_trial_visits
            #and the first post_trial_visit is within 1 month of the trial_end_time, then also include that post_trial_visit
        if len(subject.post_trial_visits) > 0:
            if np.abs((subject.post_trial_visits[0].visit_date - trial_end_time).days) <= 30:
                subject_status['date'].append(subject.post_trial_visits[0].visit_date)
                subject_status['hasSz'].append(adjust_hasSz_by_elo(subject.post_trial_visits[0], trial_start_time))

        #handle visits with an IDK classification
        subject_status = pd.DataFrame(subject_status).replace(2, idk_behavior)
        if pd.isnull(idk_behavior):
            subject_status = subject_status.dropna(subset='hasSz')
            if len(subject_status) < 1:
                pat_idx_empty_by_idk.append(i)
                continue

        trial_table['MRN'][i] = subject.patient.pat_id
        
        #if they started with seizures, they also have 0 duration and 1 event.
        if subject_status['hasSz'].iloc[0] == 1:
            #CHANGED-04162024: Instead of broadcasting only the first visit backwards to time 0 (worst case), we now assume the seizure happens on the visit date
            #trial_table['duration'][i] = 0
            trial_table['duration'][i] = (subject_status['date'].iloc[0] - trial_start_time).days
            trial_table['event'][i] = True
        #otherwise, the patient must be seizure free on the drug. Find out for how long. 
        else:
            hasSz_idx = np.argwhere(subject_status['hasSz'].to_numpy() == 1).flatten()
            
            #if there are hasSz after the szFree, then get the first index
            if len(hasSz_idx) > 0:
                next_hasSz_idx = np.min(hasSz_idx)                    
                trial_table['duration'][i] = (subject_status['date'].iloc[next_hasSz_idx] - trial_start_time).days
                trial_table['event'][i] = True
            #otherwise the patient never had another seizure, and will be censored
            else:
                trial_table['duration'][i] = (subject_status['date'].iloc[-1] - trial_start_time).days
                trial_table['event'][i] = False
        
        #check if we've crossed the max duration and censor accordingly
        if max_duration_days != None:
            if trial_table['duration'][i] > max_duration_days:
                trial_table['duration'][i] = max_duration_days
                trial_table['event'][i] = False

    #skip patients with IDK classifications who are still in the cohort
    if pd.isnull(idk_behavior):
        pat_idx_empty_by_idk = set(pat_idx_empty_by_idk)
        clinical_trial.cohort = [pat for idx, pat in enumerate(clinical_trial.cohort) if idx not in pat_idx_empty_by_idk]

    return pd.DataFrame(trial_table).dropna()

def sustained_freedom_survival(all_agg_pats, med_list, minimum_prior_asms, maximum_prior_asms, min_cohort_size, idk_behavior, 
                               medication_start_regex, medication_end_regex, brand_to_generic, max_duration_days=None, shadow_propensities=False,
                               use_IPTW=False, use_IPCW=False, confounder_imputation='MICE', confounder_table=None, metadata_cols=None, enforce_positivity=True):
    """Conduct the seizure freedom clinical trials with the specified parameters"""

    #set the trial up
    clinical_trials = {}
    for med in med_list:        
        new_trial = ctc.ClinicalTrial(med, 
                                      outcome_measure='sustained_freedom', 
                                      minimum_prior_asms=minimum_prior_asms, 
                                      maximum_prior_asms=maximum_prior_asms)
        enroll_sub_stat = new_trial.enroll_subjects(all_agg_pats, medication_start_regex, medication_end_regex, brand_to_generic, debug=False)
        if use_IPTW or use_IPCW or shadow_propensities:
            imputed_confounder_table = impute_confounders(confounder_table, metadata_cols, confounder_imputation)
            new_trial.cohort = [subject for subject in new_trial.cohort if subject.determine_confounders(imputed_confounder_table, metadata_cols) == 1]
        if len(new_trial.cohort) >= min_cohort_size:
            clinical_trials[med] = new_trial

    #Create KM trial tables
    trial_tables = {}
    for med in clinical_trials:
        trial_tables[med] = construct_survival_table_szFreedom(clinical_trials[med], idk_behavior,  max_duration_days)

    #calculate propensitity scores
    if use_IPTW or use_IPCW:
        gb_models, trial_confounders = calculate_propensity_scores(clinical_trials, use_IPTW, use_IPCW, trial_tables, max_duration_days)
        if enforce_positivity:
            trial_confounders, clinical_trials, trial_tables = enforce_positivity_on_subjects(trial_confounders, clinical_trials, trial_tables,  enforce_IPTW=use_IPTW, enforce_IPCW=use_IPCW)
    elif shadow_propensities:
        gb_models, trial_confounders = calculate_propensity_scores(clinical_trials, True, True, trial_tables, max_duration_days)
        if enforce_positivity:
            trial_confounders, clinical_trials, trial_tables = enforce_positivity_on_subjects(trial_confounders, clinical_trials, trial_tables, enforce_IPTW=True, enforce_IPCW=True)
        
    #for each trial, create and fit the KM analysis
    km_analyses = {}
    for med in clinical_trials:
        km_model = KaplanMeierFitter()
        if use_IPTW or use_IPCW:
            #calculate weights, and truncate to be between 1st and 99th percentiles
            weights = calculate_propensity_weights(clinical_trials[med])
            trial_tables[med] = trial_tables[med].merge(weights, on='MRN', how='inner')
            km_model.fit(trial_tables[med]['duration'], trial_tables[med]['event'], label=med, weights=trial_tables[med]['weight'])
        else:
            km_model.fit(trial_tables[med]['duration'], trial_tables[med]['event'], label=med)
        km_analyses[med] = km_model

    if use_IPTW or use_IPCW or shadow_propensities:
        return clinical_trials, trial_tables, gb_models, trial_confounders, km_analyses
    else:
        return clinical_trials, trial_tables, km_analyses

def get_trial_statistics(clinical_trial):
    """Print basic information on the trial cohort"""
    num_pats = {}
    num_vis = {}
    num_pres = {}
    num_pat_years = {}
    num_prior_asms_summary = []
    epiType_counts = {}
    
    for med in clinical_trial:
        #calculate trial size values
        num_pats[med] = len(clinical_trial[med].cohort)
        num_vis[med] = np.sum([len(subject.patient.aggregate_visits) for subject in clinical_trial[med].cohort])
        num_pres[med] = np.sum([len(subject.patient.medications[med].prescriptions) for subject in clinical_trial[med].cohort])
        num_pat_years[med] = np.sum([(subject._sorted_visit_dates[-1] - subject._sorted_visit_dates[0]).days for subject in clinical_trial[med].cohort])/365

        #now we count epilepsy types
        epiType_counts[med] = {}
        for subject in clinical_trial[med].cohort:
            if subject.patient.epiType not in epiType_counts[med]:
                epiType_counts[med][subject.patient.epiType] = 0
            epiType_counts[med][subject.patient.epiType] += 1

        #now we count how many prior ASMs there were
        for subject in clinical_trial[med].cohort:
            num_prior_asms_summary.append(len(subject.prior_asms))

    print("Number of patients")
    print(num_pats)
    print(f"Total: {np.sum(list(num_pats.values()))}")
    print()

    print("Number of visits")
    print(num_vis)
    print(f"Total: {np.sum(list(num_vis.values()))}")
    print()

    print("Number of prescriptions")
    print(num_pres)
    print(f"Total: {np.sum(list(num_pres.values()))}")
    print()

    print("Total patient years")
    print(num_pat_years)
    print(f"Total: {np.sum(list(num_pat_years.values()))}")
    print()

    print("Epilepsy Types")
    print(epiType_counts)
    print()

    print("Num prior ASMs")
    print(f"Mean: {np.mean(num_prior_asms_summary)}")
    print(f"Median: {np.median(num_prior_asms_summary)}")
    print(f"Stddev: {np.std(num_prior_asms_summary)}")

def get_szFreq_stats(clinical_trial, cmap, figsize=(6,4), return_vals=False, max_sz=32, kde_plot=False):
    """
    Analyze the seizure frequency of the trial cohort prior before they were enrolled to see if there is possible confounding from differences in seizure frequency
        Counts the number of seizure frequency values we get within 1 year of the starting trial date
        Returns the distribution of these values
    """
    all_szFreqs = {} #the raw values of the seizure frequencies
    all_szFreq_cts = {} #the number of seizure frequency hits
    all_median_szFreqs = {} #the median seizure frequencies
    for med in clinical_trial:
        all_szFreqs[med] = []
        #for each subject
        for subject in clinical_trial[med].cohort:
            #find their trial start date
            trial_start = subject.patient.medications[med].start_date
            subject_szFreqs = []
            #for each pre-trial visit, find those within 1 year of trial_start and add their pqfs 
            for vis in subject.pre_trial_visits:
                if (vis.visit_date - trial_start).days <= 365 and not pd.isnull(vis.pqf):
                    if vis.pqf > 0:
                        subject_szFreqs.append(vis.pqf)
            all_szFreqs[med].append(subject_szFreqs)   

        #once we have all of the szFreqs for each subject, we count how many szFreqs each one has
        all_szFreq_cts[med] = [len(sub_szFreqs) for sub_szFreqs in all_szFreqs[med]]

        #and now we calculate the median seizure frequency for each subject, capped at 30 seizures per month
        all_median_szFreqs[med] = [np.min((np.median(sub_szFreqs), max_sz)) for sub_szFreqs in all_szFreqs[med]]

    #now we plot number of szFreq values per patient
    fig = plt.figure(figsize=figsize, dpi=300)
    leg_el = []
    for med in all_szFreq_cts:
        if not kde_plot:
            plt.hist(all_szFreq_cts[med], bins=range(0,15), density=True, color=cmap[med], alpha=0.33)
        else:
            sns.kdeplot(all_szFreq_cts[med], color=cmap[med])
        leg_el.append(med)
    plt.legend(leg_el)
    plt.xlabel("Number of seizure frequency values within one year of trial start")
    plt.ylabel("Proportion of patients")
    plt.show()

    #now we plot median of szFreq values per patient
    fig = plt.figure(figsize=figsize, dpi=300)
    leg_el = []
    for med in all_median_szFreqs:
        if not kde_plot:
            plt.hist(all_median_szFreqs[med], bins=range(0,31), density=True, color=cmap[med], alpha=0.33)
        else:
            sns.kdeplot(all_median_szFreqs[med], color=cmap[med])
        leg_el.append(med)
    plt.legend(leg_el)
    plt.xlabel("Median seizure frequency per patient within one year of trial start")
    plt.ylabel("Proportion of patients")
    plt.show()

    if return_vals:
        return all_szFreqs, all_szFreq_cts, all_median_szFreqs

def get_confounder_stats(clinical_trial):
    """Prints out frequencies of covariates in the trial cohort"""
    pats_with_GTCS = {}
    pats_with_OSz = {}
    pats_with_status = {}
    pats_with_febrile = {}
    pats_with_intel = {}
    pats_with_psych = {}
    pats_with_fam = {}
    num_pats = {}

    for med in clinical_trial:
        pats_with_GTCS[med] = [subject.confounders['GTCS'] for subject in clinical_trial[med].cohort]
        pats_with_OSz[med] = [subject.confounders['other_Sz_Types'] for subject in clinical_trial[med].cohort]
        pats_with_status[med] = [subject.confounders['status_epilepticus'] for subject in clinical_trial[med].cohort]
        pats_with_febrile[med] = [subject.confounders['febrile_history'] for subject in clinical_trial[med].cohort]
        pats_with_intel[med] = [subject.confounders['intellectual_disability'] for subject in clinical_trial[med].cohort]
        pats_with_psych[med] = [subject.confounders['has_psy_com'] for subject in clinical_trial[med].cohort]
        pats_with_fam[med] = [subject.confounders['family_history'] for subject in clinical_trial[med].cohort]
        num_pats[med] = len(clinical_trial[med].cohort)

    print("Number of GTCs")
    num_pats_with_conf = {med:np.sum(pats_with_GTCS[med]) for med in pats_with_GTCS}
    perc_pats_with_conf = {med:np.mean(pats_with_GTCS[med]) for med in pats_with_GTCS}
    print(f"Number of patients: {num_pats_with_conf}")
    print(f"% of patients: {perc_pats_with_conf}")
    print(f"Total number of patients: {np.sum(list(num_pats_with_conf.values()))}")
    print(f"Total % of patients: {np.sum(list(num_pats_with_conf.values()))/ np.sum(list(num_pats.values()))}")
    print()

    print("Number of OSz")
    num_pats_with_conf = {med:np.sum(pats_with_OSz[med]) for med in pats_with_OSz}
    perc_pats_with_conf = {med:np.mean(pats_with_OSz[med]) for med in pats_with_OSz}
    print(f"Number of patients: {num_pats_with_conf}")
    print(f"% of patients: {perc_pats_with_conf}")
    print(f"Total number of patients: {np.sum(list(num_pats_with_conf.values()))}")
    print(f"Total % of patients: {np.sum(list(num_pats_with_conf.values()))/ np.sum(list(num_pats.values()))}")
    print()

    print("Number of Status")
    num_pats_with_conf = {med:np.sum(pats_with_status[med]) for med in pats_with_status}
    perc_pats_with_conf = {med:np.mean(pats_with_status[med]) for med in pats_with_status}
    print(f"Number of patients: {num_pats_with_conf}")
    print(f"% of patients: {perc_pats_with_conf}")
    print(f"Total number of patients: {np.sum(list(num_pats_with_conf.values()))}")
    print(f"Total % of patients: {np.sum(list(num_pats_with_conf.values()))/ np.sum(list(num_pats.values()))}")
    print()

    print("Number of Febrile")
    num_pats_with_conf = {med:np.sum(pats_with_febrile[med]) for med in pats_with_febrile}
    perc_pats_with_conf = {med:np.mean(pats_with_febrile[med]) for med in pats_with_febrile}
    print(f"Number of patients: {num_pats_with_conf}")
    print(f"% of patients: {perc_pats_with_conf}")
    print(f"Total number of patients: {np.sum(list(num_pats_with_conf.values()))}")
    print(f"Total % of patients: {np.sum(list(num_pats_with_conf.values()))/ np.sum(list(num_pats.values()))}")
    print()

    print("Number of Intel")
    num_pats_with_conf = {med:np.sum(pats_with_intel[med]) for med in pats_with_intel}
    perc_pats_with_conf = {med:np.mean(pats_with_intel[med]) for med in pats_with_intel}
    print(f"Number of patients: {num_pats_with_conf}")
    print(f"% of patients: {perc_pats_with_conf}")
    print(f"Total number of patients: {np.sum(list(num_pats_with_conf.values()))}")
    print(f"Total % of patients: {np.sum(list(num_pats_with_conf.values()))/ np.sum(list(num_pats.values()))}")
    print()

    print("Number of Psych")
    num_pats_with_conf = {med:np.sum(pats_with_psych[med]) for med in pats_with_psych}
    perc_pats_with_conf = {med:np.mean(pats_with_psych[med]) for med in pats_with_psych}
    print(f"Number of patients: {num_pats_with_conf}")
    print(f"% of patients: {perc_pats_with_conf}")
    print(f"Total number of patients: {np.sum(list(num_pats_with_conf.values()))}")
    print(f"Total % of patients: {np.sum(list(num_pats_with_conf.values()))/ np.sum(list(num_pats.values()))}")
    print()

    print("Number of Fam")
    num_pats_with_conf = {med:np.sum(pats_with_fam[med]) for med in pats_with_fam}
    perc_pats_with_conf = {med:np.mean(pats_with_fam[med]) for med in pats_with_fam}
    print(f"Number of patients: {num_pats_with_conf}")
    print(f"% of patients: {perc_pats_with_conf}")
    print(f"Total number of patients: {np.sum(list(num_pats_with_conf.values()))}")
    print(f"Total % of patients: {np.sum(list(num_pats_with_conf.values()))/ np.sum(list(num_pats.values()))}")
    print()

def plot_KM_curve(clinical_trials, km_analyses, plt_elements, cmap, figsize=(8,6), save_path=None, ylim=[0,0.60], xlim=None):
    """Plot the Kaplan-Meier Curves from the Clinical Trial"""
    plt.figure(figsize=figsize)
    leg_el = []
    #plot the curve
    for med in clinical_trials:
        kmf_plot = km_analyses[med].plot(ci_show=False, legend=False, c=cmap[med], linewidth=2)
        leg_el.append(f"{med} - {len(clinical_trials[med].cohort)}")
    
    #overlay just the censoring
    for med in clinical_trials:
        km_analyses[med].plot(show_censors=True, censor_styles={'ms':12, 'marker':'|', 'alpha':0.3}, ci_show=True, legend=False, linewidth=0, label='_nolegend_', c=cmap[med])
            
    plt.xlabel(plt_elements['xlabel'])
    plt.ylabel(plt_elements['ylabel'])
    plt.xticks(ticks=plt_elements['xticks'], labels=plt_elements['xtick_labels'])
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(leg_el)
    add_at_risk_counts(*[km_analyses[med] for med in clinical_trials])
    if save_path is not None:
        plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", dpi=600, bbox_inches='tight')
    
    plt.show()
    
    all_results = pairwise_log_rank(clinical_trials, km_analyses)