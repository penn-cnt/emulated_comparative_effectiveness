import numpy as np
import pandas as pd
import pickle

import pipeline_utilities as pu
import clinical_trial_classes as ctc
import clinical_trial_functions as ctf
import data_loading_functions as dlf

#initial patient and notes size
initial_epi_notes = pd.read_pickle('data/epilepsy_notes.pkl')
print(initial_epi_notes.MRN.nunique())
print(len(initial_epi_notes))

#file paths for raw data
save_fig_dir = "szFreedom_figures/"
prescription_paths = {'prescription_path':'data/medication_records', 
                      'asm_list_path':'data/ASM_list_07252023.csv', 
                      'asm_exclusion_paths':'data/exclusionary_ASM_lists.csv', 
                      'asm_usages_path':'data/ASM_usages_07252023.csv'}
metadata_path = 'data/meatadata_records.csv'
pat_path = 'data/outcome_measures.pkl'
epiType_path = 'data/epilepsy_types.pkl'

#load in patient outcomes and print number of patients without data preprocessing
with open(pat_path, 'rb') as f:
    all_agg_pats_raw = pickle.load(f)['all_agg_pats']
    print(len(all_agg_pats_raw))

#now preprocess the data
all_agg_pats, all_prescriptions, brand_to_generic, metadata, t1_asms, t2_asms, t3_asms = dlf.load_all_data(prescription_paths, metadata_path, pat_path, epiType_path)

#which patients have prescriptions?
pats_with_prescriptions = []
for pat in all_agg_pats:
    if len(pat.medications) > 1:
        pats_with_prescriptions.append(pat.pat_id)
print(len(pats_with_prescriptions))
print(len(all_agg_pats))

#set up how to get both epilepsy risk factors and semiology info
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
epi_features_regex = {'start':epi_features_start_regex, 'end':epi_features_end_regex}
semiology_regex = {'start':semiology_start_regex, 'end':semiology_end_regex}
psych_comorbidities_regex = {'start':psych_start_regex, 'end':psych_end_regex}

#get covariates
all_pat_epilepsy_features, all_pat_seizure_types, all_pat_psych_comorbidities = dlf.get_cohort_confounders(all_agg_pats, epi_features_regex, semiology_regex, psych_comorbidities_regex)
all_note_confounders = all_pat_seizure_types.merge(all_pat_epilepsy_features, on=['MRN', 'visit_date'], how='inner').drop(['semiology_text', 'feature_text'], axis=1)
all_note_confounders = all_note_confounders.merge(all_pat_psych_comorbidities, on=['MRN', 'visit_date'], how='inner').drop(['psych_text'], axis=1)
pat_metadata = metadata[['MRN', 'DOB_YR', 'CONTACT_DATE']].rename(columns={'CONTACT_DATE':'visit_date'})
all_pat_confounders = all_note_confounders.merge(pat_metadata, on=['MRN', 'visit_date'], how='inner')
all_pat_confounders = all_pat_confounders.drop_duplicates()

#how many patients have covariates and prescriptions?
print(all_pat_confounders.loc[all_pat_confounders.MRN.isin(pats_with_prescriptions)].MRN.nunique())

#now we count epilepsy types
epiType_counts = {}
for pat in all_agg_pats:
    if pat.epiType not in epiType_counts:
        epiType_counts[pat.epiType] = 0
    epiType_counts[pat.epiType] += 1
print(epiType_counts)