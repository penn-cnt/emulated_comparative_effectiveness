import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

import pipeline_utilities as pu
import clinical_trial_classes as ctc
import clinical_trial_functions as ctf
import data_loading_functions as dlf

from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

#Load in necessary data
save_fig_dir = "szFreedom_figures/"
prescription_paths = {'prescription_path':'data/medication_records', 
                      'asm_list_path':'data/ASM_list_07252023.csv', 
                      'asm_exclusion_paths':'data/exclusionary_ASM_lists.csv', 
                      'asm_usages_path':'data/ASM_usages_07252023.csv'}
metadata_path = 'data/meatadata_records.csv'
pat_path = 'data/outcome_measures.pkl'
epiType_path = 'data/epilepsy_types.pkl'
all_agg_pats, all_prescriptions, brand_to_generic, metadata, t1_asms, t2_asms, t3_asms = dlf.load_all_data(prescription_paths, metadata_path, pat_path, epiType_path)

#load regex patterns to extract epilepsy characteristics from notes
time_pattern, AS_pattern, base_med_pattern, base_asm_pattern, ASM_pattern, medication_pattern,\
        sz_pattern, seizure_desc_pattern, semiology_pattern, features_section_pattern, semiology_section_pattern,\
        type_pattern, history_pattern, study_pattern, exam_pattern, plan_pattern, hpi_pattern, other_pattern = dlf.load_section_regex()
section_pattern = rf"{exam_pattern}|{plan_pattern}|{hpi_pattern}|{other_pattern}"
semiology_start_regex = rf"(?im){semiology_section_pattern}"
semiology_end_regex = rf"(?im)({ASM_pattern})|({medication_pattern})|({features_section_pattern})|({history_pattern})|({study_pattern})|({section_pattern})"
epi_features_start_regex = rf"(?im){features_section_pattern}"
epi_features_end_regex = rf"(?im)({ASM_pattern})|({medication_pattern})|({semiology_section_pattern})|({history_pattern})|({study_pattern})|({section_pattern})"
medication_start_regex = rf"(?im)({ASM_pattern})|({medication_pattern})"
medication_end_regex = rf"(?im)({semiology_section_pattern})|({features_section_pattern})|({history_pattern})|({study_pattern})|({section_pattern})"
psych_additional_patterns = rf"(\bob/gyn\b)|(\bwork\b)|(\b(family|social) history\b)|(\bmedications\b)"
psych_start_regex = rf"(?im)past psychiatric history"
psych_end_regex = rf"(?im)({psych_additional_patterns})|({ASM_pattern})|({medication_pattern})|({features_section_pattern})|({semiology_section_pattern})|({history_pattern})|({study_pattern})|({section_pattern})"

#get epilepsy features and semiology
epi_features_regex = {'start':epi_features_start_regex, 'end':epi_features_end_regex}
semiology_regex = {'start':semiology_start_regex, 'end':semiology_end_regex}
psych_comorbidities_regex = {'start':psych_start_regex, 'end':psych_end_regex}
all_pat_epilepsy_features, all_pat_seizure_types, all_pat_psych_comorbidities = dlf.get_cohort_confounders(all_agg_pats, epi_features_regex, semiology_regex, psych_comorbidities_regex)

#get patients with both epilepsy risk factors and semiology info
all_note_confounders = all_pat_seizure_types.merge(all_pat_epilepsy_features, on=['MRN', 'visit_date'], how='inner').drop(['semiology_text', 'feature_text'], axis=1)
all_note_confounders = all_note_confounders.merge(all_pat_psych_comorbidities, on=['MRN', 'visit_date'], how='inner').drop(['psych_text'], axis=1)

#get psychiatric comorbidities 
pat_metadata = metadata[['MRN', 'DOB_YR', 'CONTACT_DATE', 'GENDER']].rename(columns={'CONTACT_DATE':'visit_date'})
pat_metadata.GENDER.replace({'F':0, 'M':1, 'X':2}, inplace=True)# map Female gender to 0, male to 1
all_pat_confounders = all_note_confounders.merge(pat_metadata, on=['MRN', 'visit_date'], how='inner')
all_pat_confounders = all_pat_confounders.drop_duplicates()

#get visit metadata
metadata_columns = ['MRN', 'visit_date', 'DOB_YR']
categorical_columns = ['GTCS', 'other_Sz_Types', 'status_epilepticus', 'febrile_history', 'intellectual_disability', 'family_history', 'has_psy_com', 'GENDER']
all_pat_confounders = dlf.convert_categorical(all_pat_confounders, categorical_columns, categorical_columns)

#set up trial info
min_cohort_size = 100
max_duration_days = 365*2
idk_behavior=np.nan #ignores visits that do not have a seizure freedom classification
cmap = dlf.get_cmap()
plt_elements = {
    'xlabel':"Years After Starting Drug",
    'ylabel':"Proportion of Patients Seizure Free",
    'xtick_labels':np.arange(0, max_duration_days+1, 182.5)/365,
    'xticks':np.arange(0, max_duration_days+1, 182.5),
}

#run primary trials
clinical_trials_L1, trial_tables_L1, gb_models_L1, trial_confounders_L1, km_analyses_L1 = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                          med_list=t1_asms, 
                                                                                                          minimum_prior_asms=None, 
                                                                                                          maximum_prior_asms=None, 
                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                          idk_behavior=idk_behavior, 
                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                          max_duration_days=max_duration_days,
                                                                                                          use_IPTW=True, use_IPCW=True, 
                                                                                                          confounder_imputation='mice',
                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                          metadata_cols=metadata_columns)
clinical_trials_L2, trial_tables_L2, gb_models_L2, trial_confounders_L2, km_analyses_L2 = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                          med_list=t2_asms, 
                                                                                                          minimum_prior_asms=None, 
                                                                                                          maximum_prior_asms=None, 
                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                          idk_behavior=idk_behavior, 
                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                          max_duration_days=max_duration_days,
                                                                                                          use_IPTW=True, use_IPCW=True, 
                                                                                                          confounder_imputation='mice',
                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                          metadata_cols=metadata_columns)
clinical_trials_L3, trial_tables_L3, gb_models_L3, trial_confounders_L3, km_analyses_L3 = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                          med_list=t3_asms, 
                                                                                                          minimum_prior_asms=None, 
                                                                                                          maximum_prior_asms=None, 
                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                          idk_behavior=idk_behavior, 
                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                          max_duration_days=max_duration_days,
                                                                                                          use_IPTW=True, use_IPCW=True, 
                                                                                                          confounder_imputation='mice',
                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                          metadata_cols=metadata_columns)

#print out trial characteristics
ctf.get_trial_statistics(clinical_trials_L1)
ctf.get_trial_statistics(clinical_trials_L2)
ctf.get_trial_statistics(clinical_trials_L3)
ctf.get_confounder_stats(clinical_trials_L1)
ctf.get_confounder_stats(clinical_trials_L2)
ctf.get_confounder_stats(clinical_trials_L3)

#baseline seizure frequency analysis - is there confounding as a result of seizure frequency?
ctf.get_szFreq_stats(clinical_trials_L1, cmap=cmap, kde_plot=True)
ctf.get_szFreq_stats(clinical_trials_L2, cmap=cmap, kde_plot=True)
ctf.get_szFreq_stats(clinical_trials_L3, cmap=cmap, kde_plot=True)

#plot kaplan meier analyses results
ctf.plot_KM_curve(clinical_trials_L1, km_analyses_L1, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L1_KM')
ctf.plot_KM_curve(clinical_trials_L2, km_analyses_L2, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L2_KM')
ctf.plot_KM_curve(clinical_trials_L3, km_analyses_L3, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L3_KM')

#assess positivity of propensity scores in each trial
ctf.plot_iptw_positivity(trial_confounders_L1, main_save_dir=save_fig_dir)
ctf.plot_ipcw_positivity(trial_confounders_L1, main_save_dir=save_fig_dir)
ctf.plot_iptw_positivity(trial_confounders_L2, main_save_dir=save_fig_dir)
ctf.plot_ipcw_positivity(trial_confounders_L2, main_save_dir=save_fig_dir)
ctf.plot_iptw_positivity(trial_confounders_L3, main_save_dir=save_fig_dir)
ctf.plot_ipcw_positivity(trial_confounders_L3, main_save_dir=save_fig_dir)

#assess bias balance before and after weighting in each trial
ctf.plot_pre_post_weighting_bias_balance_iptw(trial_confounders_L1, main_save_dir=save_fig_dir)
ctf.plot_pre_post_weighting_bias_balance_ipcw(trial_confounders_L1, main_save_dir=save_fig_dir)
ctf.plot_pre_post_weighting_bias_balance_iptw(trial_confounders_L2, main_save_dir=save_fig_dir)
ctf.plot_pre_post_weighting_bias_balance_ipcw(trial_confounders_L2, main_save_dir=save_fig_dir)
ctf.plot_pre_post_weighting_bias_balance_iptw(trial_confounders_L3, main_save_dir=save_fig_dir)
ctf.plot_pre_post_weighting_bias_balance_ipcw(trial_confounders_L3, main_save_dir=save_fig_dir)

# ======================================= END PRIMARY ANALYSIS ======================================= #

#Sensitivity analysis - no imputation
clinical_trials_L1_noImpute, trial_tables_L1_noImpute, gb_models_L1_noImpute, trial_confounders_L1_noImpute, km_analyses_L1_noImpute = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                                          med_list=t1_asms, 
                                                                                                                                          minimum_prior_asms=None, 
                                                                                                                                          maximum_prior_asms=None, 
                                                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                                                          idk_behavior=idk_behavior, 
                                                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                                                          max_duration_days=max_duration_days,
                                                                                                                                          use_IPTW=True, use_IPCW=True, 
                                                                                                                                          confounder_imputation=None,
                                                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                                                          metadata_cols=metadata_columns)
clinical_trials_L2_noImpute, trial_tables_L2_noImpute, gb_models_L2_noImpute, trial_confounders_L2_noImpute, km_analyses_L2_noImpute = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                                          med_list=t2_asms, 
                                                                                                                                          minimum_prior_asms=None, 
                                                                                                                                          maximum_prior_asms=None, 
                                                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                                                          idk_behavior=idk_behavior, 
                                                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                                                          max_duration_days=max_duration_days,
                                                                                                                                          use_IPTW=True, use_IPCW=True, 
                                                                                                                                          confounder_imputation=None,
                                                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                                                          metadata_cols=metadata_columns)
clinical_trials_L3_noImpute, trial_tables_L3_noImpute, gb_models_L3_noImpute, trial_confounders_L3_noImpute, km_analyses_L3_noImpute = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                                          med_list=t3_asms, 
                                                                                                                                          minimum_prior_asms=None, 
                                                                                                                                          maximum_prior_asms=None, 
                                                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                                                          idk_behavior=idk_behavior, 
                                                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                                                          max_duration_days=max_duration_days,
                                                                                                                                          use_IPTW=True, use_IPCW=True, 
                                                                                                                                          confounder_imputation=None,
                                                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                                                          metadata_cols=metadata_columns)

#plot kaplan meier results
ctf.plot_KM_curve(clinical_trials_L1_noImpute, km_analyses_L1_noImpute, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L1_noImpute_KM')
ctf.plot_KM_curve(clinical_trials_L2_noImpute, km_analyses_L2_noImpute, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L2_noImpute_KM')
ctf.plot_KM_curve(clinical_trials_L3_noImpute, km_analyses_L3_noImpute, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L3_noImpute_KM')

#Sensitivity analysis - no IPCW
clinical_trials_L1_noIPCW, trial_tables_L1_noIPCW, gb_models_L1_noIPCW, trial_confounders_L1_noIPCW, km_analyses_L1_noIPCW = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                              med_list=t1_asms, 
                                                                                                                              minimum_prior_asms=None, 
                                                                                                                              maximum_prior_asms=None, 
                                                                                                                              min_cohort_size=min_cohort_size, 
                                                                                                                              idk_behavior=idk_behavior, 
                                                                                                                              medication_start_regex=medication_start_regex,
                                                                                                                              medication_end_regex=medication_end_regex,
                                                                                                                              brand_to_generic=brand_to_generic,
                                                                                                                              max_duration_days=max_duration_days,
                                                                                                                              use_IPTW=True, use_IPCW=False, 
                                                                                                                              confounder_imputation='mice',
                                                                                                                              confounder_table=all_pat_confounders, 
                                                                                                                              metadata_cols=metadata_columns)
clinical_trials_L2_noIPCW, trial_tables_L2_noIPCW, gb_models_L2_noIPCW, trial_confounders_L2_noIPCW, km_analyses_L2_noIPCW = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                              med_list=t2_asms, 
                                                                                                                              minimum_prior_asms=None, 
                                                                                                                              maximum_prior_asms=None, 
                                                                                                                              min_cohort_size=min_cohort_size, 
                                                                                                                              idk_behavior=idk_behavior, 
                                                                                                                              medication_start_regex=medication_start_regex,
                                                                                                                              medication_end_regex=medication_end_regex,
                                                                                                                              brand_to_generic=brand_to_generic,
                                                                                                                              max_duration_days=max_duration_days,
                                                                                                                              use_IPTW=True, use_IPCW=False, 
                                                                                                                              confounder_imputation='mice',
                                                                                                                              confounder_table=all_pat_confounders, 
                                                                                                                              metadata_cols=metadata_columns)
clinical_trials_L3_noIPCW, trial_tables_L3_noIPCW, gb_models_L3_noIPCW, trial_confounders_L3_noIPCW, km_analyses_L3_noIPCW = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                              med_list=t3_asms, 
                                                                                                                              minimum_prior_asms=None, 
                                                                                                                              maximum_prior_asms=None, 
                                                                                                                              min_cohort_size=min_cohort_size, 
                                                                                                                              idk_behavior=idk_behavior, 
                                                                                                                              medication_start_regex=medication_start_regex,
                                                                                                                              medication_end_regex=medication_end_regex,
                                                                                                                              brand_to_generic=brand_to_generic,
                                                                                                                              max_duration_days=max_duration_days,
                                                                                                                              use_IPTW=True, use_IPCW=False, 
                                                                                                                              confounder_imputation='mice',
                                                                                                                              confounder_table=all_pat_confounders, 
                                                                                                                              metadata_cols=metadata_columns)

#plot kaplan meier results
ctf.plot_KM_curve(clinical_trials_L1_noIPCW, km_analyses_L1_noIPCW, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L1_noIPCW_KM')
ctf.plot_KM_curve(clinical_trials_L2_noIPCW, km_analyses_L2_noIPCW, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L2_noIPCW_KM')
ctf.plot_KM_curve(clinical_trials_L3_noIPCW, km_analyses_L3_noIPCW, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L3_noIPCW_KM')

#Sensitivity analysis - no IPTW
clinical_trials_L1_noIPTW, trial_tables_L1_noIPTW, gb_models_L1_noIPTW, trial_confounders_L1_noIPTW, km_analyses_L1_noIPTW = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                              med_list=t1_asms, 
                                                                                                                              minimum_prior_asms=None, 
                                                                                                                              maximum_prior_asms=None, 
                                                                                                                              min_cohort_size=min_cohort_size, 
                                                                                                                              idk_behavior=idk_behavior, 
                                                                                                                              medication_start_regex=medication_start_regex,
                                                                                                                              medication_end_regex=medication_end_regex,
                                                                                                                              brand_to_generic=brand_to_generic,
                                                                                                                              max_duration_days=max_duration_days,
                                                                                                                              use_IPTW=False, use_IPCW=True, 
                                                                                                                              confounder_imputation='mice',
                                                                                                                              confounder_table=all_pat_confounders, 
                                                                                                                              metadata_cols=metadata_columns)
clinical_trials_L2_noIPTW, trial_tables_L2_noIPTW, gb_models_L2_noIPTW, trial_confounders_L2_noIPTW, km_analyses_L2_noIPTW = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                              med_list=t2_asms, 
                                                                                                                              minimum_prior_asms=None, 
                                                                                                                              maximum_prior_asms=None, 
                                                                                                                              min_cohort_size=min_cohort_size, 
                                                                                                                              idk_behavior=idk_behavior, 
                                                                                                                              medication_start_regex=medication_start_regex,
                                                                                                                              medication_end_regex=medication_end_regex,
                                                                                                                              brand_to_generic=brand_to_generic,
                                                                                                                              max_duration_days=max_duration_days,
                                                                                                                              use_IPTW=False, use_IPCW=True, 
                                                                                                                              confounder_imputation='mice',
                                                                                                                              confounder_table=all_pat_confounders, 
                                                                                                                              metadata_cols=metadata_columns)
clinical_trials_L3_noIPTW, trial_tables_L3_noIPTW, gb_models_L3_noIPTW, trial_confounders_L3_noIPTW, km_analyses_L3_noIPTW = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                                              med_list=t3_asms, 
                                                                                                                              minimum_prior_asms=None, 
                                                                                                                              maximum_prior_asms=None, 
                                                                                                                              min_cohort_size=min_cohort_size, 
                                                                                                                              idk_behavior=idk_behavior, 
                                                                                                                              medication_start_regex=medication_start_regex,
                                                                                                                              medication_end_regex=medication_end_regex,
                                                                                                                              brand_to_generic=brand_to_generic,
                                                                                                                              max_duration_days=max_duration_days,
                                                                                                                              use_IPTW=False, use_IPCW=True, 
                                                                                                                              confounder_imputation='mice',
                                                                                                                              confounder_table=all_pat_confounders, 
                                                                                                                              metadata_cols=metadata_columns)

#plot kaplan meier results
ctf.plot_KM_curve(clinical_trials_L1_noIPTW, km_analyses_L1_noIPTW, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L1_noIPTW_KM')
ctf.plot_KM_curve(clinical_trials_L2_noIPTW, km_analyses_L2_noIPTW, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L2_noIPTW_KM')
ctf.plot_KM_curve(clinical_trials_L3_noIPTW, km_analyses_L3_noIPTW, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L3_noIPTW_KM')

#Sensitivity analysis - no propensity scoring
clinical_trials_L1_noProp, trial_tables_L1_noProp, gb_models_L1_noProp, trial_confounders_L1_noProp, km_analyses_L1_noProp = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                          med_list=t1_asms, 
                                                                                                          minimum_prior_asms=None, 
                                                                                                          maximum_prior_asms=None, 
                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                          idk_behavior=idk_behavior, 
                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                          max_duration_days=max_duration_days,
                                                                                                          shadow_propensities=True,
                                                                                                          use_IPTW=False, use_IPCW=False, 
                                                                                                          confounder_imputation='mice',
                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                          metadata_cols=metadata_columns)
clinical_trials_L2_noProp, trial_tables_L2_noProp, gb_models_L2_noProp, trial_confounders_L2_noProp, km_analyses_L2_noProp = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                          med_list=t2_asms, 
                                                                                                          minimum_prior_asms=None, 
                                                                                                          maximum_prior_asms=None, 
                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                          idk_behavior=idk_behavior, 
                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                          max_duration_days=max_duration_days,
                                                                                                          shadow_propensities=True,
                                                                                                          use_IPTW=False, use_IPCW=False, 
                                                                                                          confounder_imputation='mice',
                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                          metadata_cols=metadata_columns)
clinical_trials_L3_noProp, trial_tables_L3_noProp, gb_models_L3_noProp, trial_confounders_L3_noProp, km_analyses_L3_noProp = ctf.sustained_freedom_survival(all_agg_pats, 
                                                                                                          med_list=t3_asms, 
                                                                                                          minimum_prior_asms=None, 
                                                                                                          maximum_prior_asms=None, 
                                                                                                          min_cohort_size=min_cohort_size, 
                                                                                                          idk_behavior=idk_behavior, 
                                                                                                          medication_start_regex=medication_start_regex,
                                                                                                          medication_end_regex=medication_end_regex,
                                                                                                          brand_to_generic=brand_to_generic,
                                                                                                          max_duration_days=max_duration_days,
                                                                                                          shadow_propensities=True,
                                                                                                          use_IPTW=False, use_IPCW=False, 
                                                                                                          confounder_imputation='mice',
                                                                                                          confounder_table=all_pat_confounders, 
                                                                                                          metadata_cols=metadata_columns)

#plot kaplan meier results
ctf.plot_KM_curve(clinical_trials_L1_noProp, km_analyses_L1_noProp, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L1_noProp_KM')
ctf.plot_KM_curve(clinical_trials_L2_noProp, km_analyses_L2_noProp, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L2_noProp_KM')
ctf.plot_KM_curve(clinical_trials_L3_noProp, km_analyses_L3_noProp, plt_elements, cmap, ylim=[0,1.1], save_path=f'{save_fig_dir}/L3_noProp_KM')