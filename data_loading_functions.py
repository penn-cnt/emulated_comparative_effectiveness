import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import regex as re
from copy import deepcopy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
from lifelines.plotting import add_at_risk_counts
import itertools

def cast_to_int(x, return_original=False):
    """Converts a variable to int"""
    try:
        return int(x)
    except:
        if return_original:
            return x
        else:
            return np.nan

def cast_to_float(x, return_original=False):
    """Converts a variable to float"""
    try:
        return float(x)
    except:
        if return_original:
            return x
        else:
            return np.nan

def get_cmap(asm_list=None):
    return {
        'lamotrigine': '#E69F00',
        'levetiracetam': '#0072B2',
        'oxcarbazepine': '#009E73',
        'lacosamide': '#E69F00',
        'zonisamide': '#0072B2',
        'topiramate': '#009E73',
        'cenobamate': '#0231AB',
        'valproic acid': '#785EF0',
        'brivaracetam': '#BB1868',
        'pregabalin': '#FE6100',
        'clobazam': '#D49407',
        'carbamazepine':'#214d4e',
        'perampanel':'#64d4fd'
    }

def str_to_float(s):
    """Converts a string to float"""
    try:
        return float(s)
    except:
        return np.nan

def convert_categorical(df, columns, negative_to_nans):
    """Converts columns in a df to categorical dtypes"""
    for column in columns:
        if column in negative_to_nans:
            try:
                df[column] = df[column].clip(lower=-0.1)
                df[column] = df[column].replace(to_replace=-0.1, value=np.nan)
            except Exception as e:
                print(f"{column} | {e}")
        df[column] = df[column].astype('category')
    return df

#import additional classes. We had to put these imports here otherwise the import interdependencies become a problem
import pipeline_utilities as pu
import clinical_trial_classes as ctc
import clinical_trial_functions as ctf

def get_prior_ASMs_from_note(prior_asm_txt, brand_to_generic, return_txt = False):
    """
    Find what ASMs a patient is taking using a section of a note
    """
    prior_asm_txt = re.sub(r'\W+', ' ', prior_asm_txt) #replace all non alphanumeric characters with spaces. 
    prior_asm_txt = re.sub(r'\bper\b', '__per__', prior_asm_txt) #replace all "per" with a flag
    prior_asm_txt = prior_asm_txt.lower()
    prior_asm_set = set([brand_to_generic[word] for word in prior_asm_txt.split() if not pd.isnull(brand_to_generic[word])])
    if return_txt:
        return prior_asm_txt, prior_asm_set
    else:
        return prior_asm_set

def add_prior_ASMs(pat, prior_asm_set):
    """
    Adds prior ASMs to a patient
    """
    #calculate the earliest visit for this patient
    earliest_visit_date = np.min([vis.visit_date for vis in pat.aggregate_visits])
    pat.medications['prior_asms'] = set(prior_asm_set)
    
    #check if any of this patient's prescriptions started before the earliest visit_date, in which case they are prior asms
    for asm_name in pat.medications:
        if asm_name == 'prior_asms':
            continue
        if (pat.medications[asm_name].start_date < earliest_visit_date):
            pat.medications['prior_asms'].add(asm_name)

def add_medications_to_pats(i, all_agg_pats, all_prescriptions, brand_to_generic, medication_start_regex, medication_end_regex):   
    """
    Add medications to all patients
    """
    pat = all_agg_pats[i]
    
    #change pat.medications from an empty list to a dict with one empty set
    pat.medications = {'prior_asms':set()}
    #get patient prescriptions
    pat_prescriptions = all_prescriptions.loc[all_prescriptions.MRN == pat.pat_id]

    #for each prescription 
    for idx, row in pat_prescriptions.iterrows():

        #if this med_name could not be parsed, then skip it
        if pd.isnull(row.NAME):
            continue

        #create a new PrescriptionRecord using this information
        prescriptionRecord = ctc.PrescriptionRecord(pat, row.NAME, row.ORDERING_DATE, row.START_DATE, row.END_DATE)
        #create a new Prescription using this information
        prescription = ctc.Prescription(pat, row.NAME, row.DESCRIPTION, row.ORDERING_DATE, row.START_DATE, row.END_DATE, row.HV_DISCRETE_DOSE, row.DOSE_UNIT, row.FREQUENCY)
        #add this Prescription to the PrescriptionRecord
        prescriptionRecord.add_Prescription(prescription)

        #if this record already exists, update it
        if row.NAME in pat.medications:
            pat.medications[row.NAME].update_PrescriptionRecord(prescriptionRecord)
        #otherwise create an entry in pat.medications for this medication
        else:
            pat.medications[row.NAME] = prescriptionRecord

    #once prescriptions have been added, let's look for prior ASMSs in their very virst visit
    #get the prior asm text in the first visits' text
    prior_asm_txt = get_note_section(sorted(pat.aggregate_visits, key=lambda x: x.visit_date)[0].full_text, medication_start_regex, medication_end_regex)
    
    #find all mentions of medications in the medication section and add them as prior asms
    add_prior_ASMs(pat, get_prior_ASMs_from_note(prior_asm_txt, brand_to_generic))

    return i, pat 

def load_prescription_data(prescription_path, asm_list_path, asm_exclusion_paths, asm_usages_path=None):
    """Loads prescription data from files"""
    #load medications.
    all_prescriptions = pd.read_pickle(prescription_path)
    #drop duplicated entries and keep only outpatient all_prescriptions
    all_prescriptions = all_prescriptions.drop_duplicates(subset=all_prescriptions.columns[:-3])
    all_prescriptions = all_prescriptions.loc[all_prescriptions.ORDER_MODE != 'Inpatient']

    if asm_usages_path != None:
        #what medications are rescue medications(1), which are ASMs(0), and which aren't useful to us (2)
        med_classes = pd.read_csv(asm_usages_path, index_col=0)
        asm_generics = set(med_classes.loc[med_classes['class'] == 0].index)
        rescue_generics = set(med_classes.loc[med_classes['class']==1].index)

        #get medication names from their descriptions
        all_prescriptions['NAME'], brand_to_generic = pu.get_all_asm_names_from_description(asm_list_path, all_prescriptions, 'DESCRIPTION', 
                                                                                            asm_subset=asm_generics, path_to_exclusion_names=asm_exclusion_paths,
                                                                                            return_name_dict=True)
        
        #we want only ASMs
        all_prescriptions = all_prescriptions.loc[all_prescriptions.NAME.isin(asm_generics)]
        
        #set ASMs into tiers
        t1_asms = ['levetiracetam', 'lamotrigine', 'oxcarbazepine']
        t2_asms = ['lacosamide', 'topiramate', 'zonisamide']
        t3_asms = set(all_prescriptions.NAME.unique()) - set(t1_asms) - set(t2_asms)

        return all_prescriptions, brand_to_generic, t1_asms, t2_asms, t3_asms
    
    else:
        #get medication names from their descriptions
        all_prescriptions['NAME'], brand_to_generic = pu.get_all_asm_names_from_description(asm_list_path, all_prescriptions, 'DESCRIPTION', 
                                                                                            path_to_exclusion_names=asm_exclusion_paths, return_name_dict=True)
        return all_prescriptions, brand_to_generic

def check_epilepsy_ICD_codes(val):
    """Look for ICD10 F,G, and ICD9 290-390"""
    if isinstance(val, str):
        if 'F' in val:
            return True
        if 'G' in val:
            return True
        for i in range(290, 390):
            if str(i) in val:
                return True
        return False
    elif isinstance(val, set):
        for code in val:
            if 'F' in str(code) or 'G' in str(code):
                return True
            for i in range(290, 390):
                if str(i) in str(code):
                    return True
        return False

def load_metadata(metadata_path):
    """Load metadata from files"""
    #load in the metadata information
    if '.pkl' in metadata_path:
        metadata = pd.read_pickle(metadata_path)
    elif '.xlsx' in metadata_path:
        metadata = pd.read_excel(metadata_path)
    else:
        metadata = pd.read_csv(metadata_path)
        
    #look specifically for patients with epilepsy ICD codes
    metadata = metadata.dropna(subset='DX_CODE')
    metadata = metadata.loc[metadata.DX_CODE.apply(check_epilepsy_ICD_codes)]
    
    #format data
    metadata.MRN = metadata.MRN.apply(lambda x: str(x).rjust(9, '0'))
    metadata.CONTACT_DATE = metadata.CONTACT_DATE.apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))
    return metadata

def load_all_data(prescription_paths, metadata_path, pat_path, epiType_path):
    """Load all data for clinical trials at once"""
    #load prescription data
    if prescription_paths['asm_usages_path'] != None:
        all_prescriptions, brand_to_generic, t1_asms, t2_asms, t3_asms = load_prescription_data(
                                                                            prescription_paths['prescription_path'], 
                                                                            prescription_paths['asm_list_path'], 
                                                                            prescription_paths['asm_exclusion_paths'], 
                                                                            prescription_paths['asm_usages_path'])
    else:
        all_prescriptions, brand_to_generic = load_prescription_data(
                                                prescription_paths['prescription_path'], 
                                                prescription_paths['asm_list_path'], 
                                                prescription_paths['asm_exclusion_paths'])

    #load metadata
    metadata = load_metadata(metadata_path)

    #load regex
    time_pattern, AS_pattern, base_med_pattern, base_asm_pattern, ASM_pattern, medication_pattern,\
        sz_pattern, seizure_desc_pattern, semiology_pattern, features_section_pattern, semiology_section_pattern,\
        type_pattern, history_pattern, study_pattern, exam_pattern, plan_pattern, hpi_pattern, other_pattern = load_section_regex()
    section_pattern = rf"{exam_pattern}|{plan_pattern}|{hpi_pattern}|{other_pattern}"
    semiology_start_regex = rf"(?im){semiology_section_pattern}"
    semiology_end_regex = rf"(?im)({ASM_pattern})|({medication_pattern})|({features_section_pattern})|({history_pattern})|({study_pattern})|({section_pattern})"
    epi_features_start_regex = rf"(?im){features_section_pattern}"
    epi_features_end_regex = rf"(?im)({ASM_pattern})|({medication_pattern})|({semiology_section_pattern})|({history_pattern})|({study_pattern})|({section_pattern})"
    medications_start_regex = rf"(?im)({ASM_pattern})|({medication_pattern})"
    medications_end_regex = rf"(?im)({semiology_section_pattern})|({features_section_pattern})|({history_pattern})|({study_pattern})|({section_pattern})"
    
    #load notes and outcomes
    with open(pat_path, 'rb') as f:
        all_agg_pats = pickle.load(f)['all_agg_pats']

    #add medications to patients
    all_agg_pats = [add_medications_to_pats(i, all_agg_pats, all_prescriptions, brand_to_generic, medications_start_regex, medications_end_regex)[1] for i in range(len(all_agg_pats))]

    #add epilepsy types to the patients
    epiTypes = pd.read_pickle(epiType_path)
    epiTypes['note_len'] = epiTypes.NOTE_TEXT.apply(lambda x: len(x))
    epiTypes = epiTypes.loc[epiTypes.note_len > 750]

    #associate an epilepsy type with each patient
    for idx in range(len(all_agg_pats)):
        pat = all_agg_pats[idx]
        #find this patient's classifications of epilepsy type
        #pick the most common one that is not unknown
        this_pat_types = epiTypes.loc[epiTypes.MRN == pat.pat_id]
        epiType_cts = this_pat_types.epiType_prediction.value_counts()
        if 'Other' in epiType_cts:
            epiType_cts = epiType_cts.drop('Other')
        if len(epiType_cts) <= 0:
            pat.epiType = 'Other'
        else:
            pat.epiType = epiType_cts.idxmax()

    if prescription_paths['asm_usages_path'] != None:
        return all_agg_pats, all_prescriptions, brand_to_generic, metadata, t1_asms, t2_asms, t3_asms
    else:
        return all_agg_pats, all_prescriptions, brand_to_generic, metadata

def load_section_regex():
    """Wrapper to load note section regex"""
    return pu.load_section_regex()

def get_note_section(txt, start_regex, end_regex, splitter="    "):
    """Get note sections, separated by a '|'"""
    return "| ".join(pu.get_section_from_note(txt, start_regex, end_regex, splitter)).strip()

def identify_seizures_in_semiology(txt, semiology_start_regex, take_longest_instance=True):
    """Find what types of seizures a patient has in their semiology"""
    if take_longest_instance:
        txt = max(txt.split("| "), key=len)

    seizure_types = {"semiology_text":txt, "GTCS":False, "other_Sz_Types":False}
    
    #split the string into structured statements of event descriptions
    integer_regex = r"(?<!\.)[0-9]+(?!\.[0-9]+)" #https://www.reddit.com/r/regex/comments/10sj3bj/how_to_exclude_floating_pointdecimal_numbers/
    event_split_regex = rf"(?im)(?:(?:seizure|event)(?: type)? ?\#?)?{integer_regex}(?:\.|:|\))"

    #if event_split_regex is not there, then replace semiology_start_regex with "", as otherewise, this string will start the subtxt
    if not re.search(event_split_regex, txt):
        txt = re.sub(f"({semiology_start_regex}):?", "", txt).strip()
    event_descriptions = re.split(event_split_regex, txt)

    #after splitting, we may need to remove the first index, as this is typically the section header
    if len(event_descriptions) > 1 and re.search(semiology_start_regex, event_descriptions[0]):
        del event_descriptions[0]
    
    #regex to identify mentions of seizures
    tcs_regex = r"(?im)(((generalized|((focal( |-)to( |-))?bilateral))( |-))?tonic( |-)clonic( seizure)?)|GTC|TCSz?|F?BTCS?z?|Grand Mal|Convulsions?|generalize(s|d)?"

    #regex to identify negated mentions of seizures
    negation_regex = rf"(?im)(no (clear )?TCS_REGEX)|TCS_REGEX:( ?)no"
    
    #search for mentions of GTCs and replace these with its regex placeholder strings
    #then, collapse consecutively repeated mentions into one
    all_descriptions_processed = []
    for sub_txt in event_descriptions:
        sub_txt = sub_txt.strip()
        sub_txt = re.sub(tcs_regex, 'TCS_REGEX', sub_txt)
        sub_txt = re.sub(r"(TCS_REGEX ?){2,}", 'TCS_REGEX', sub_txt)
        
        #search for any matches to the negation regex and replace them with a negation placeholder
        sub_txt = re.sub(negation_regex, 'NEGATION_REGEX', sub_txt)

        #check if any of the seizure mentions were not negated, if so, the patient likely had that seizure type
        if 'TCS_REGEX' in sub_txt:
            seizure_types["GTCS"] = True
        #if no GTCs were found, then check if the the passage is empty or not. 
        #If it isn't, check if it says only "as noted above/below".
        #If not, default to other_Sz_Types
        elif len(sub_txt) > 1 and not (re.search(r"(?im)(above)|(below)", sub_txt) and len(sub_txt) <= 15):
            seizure_types["other_Sz_Types"] = True 
        all_descriptions_processed.append(sub_txt)

    return seizure_types, all_descriptions_processed

def get_feature_from_explicit_answer(subtxt, section_header_first_word_regex=r"(?im)status|self|precipitating|epilepsy|abnormal|febrile|CNS|intellectual|mental|cognitive|cerebral|head|neurosurgical|stroke|alcohol|drug|family|history"):
    """What features are explicitly written in the note following a rough note template?"""
    #section_header_first_word_regex - this variable dictates what explicit subsections are within the section, For example "status epilepticus: ... febrile seizures: ..."
    value_regex = r"(?im): {0,3}\b[a-zA-Z]+"
    no_regex = r"(?im)(\bno)|(\bdenies)|(\bdeny)"
    unknown_regex = r"(?im)unknown|unclear|uncertain|unlikely|unaware"

    #get the value of the feature
    value = re.findall(value_regex, subtxt)

    #for safety check if any value was found.
    if len(value) == 0:
        return -1
    else:
        value = value[0]
        
    #first, check if it looks like it's going to the next section header
    #if so, return nan
    if re.search(section_header_first_word_regex, value):
        return -2

    #next, check if it's a "no" or "none" or similar
    if re.search(no_regex, value):
        return False

    #next, we check if it's unknown
    if re.search(unknown_regex, value):
        return -3

    #finally, check if it's blank
    if not re.search(r"(?im)\b[a-zA-Z]+", value):
        return -4

    #otherwise, it must be True
    return True

def get_feature_from_verbose_answer(sentence, negative_regex = None, positive_regex = None):
    """What features are written in a note freehand?"""
    negative_matches = []
    positive_matches = []

    #search for negative regex strings in our sentence
    if negative_regex is not None:
        #if only one negative regex is passed,
        if isinstance(negative_regex, str):
            negative_matches.append(re.search(negative_regex, sentence)) 
        #if multiple negative regex are passed,
        else:
            negative_matches = [bool(re.search(neg_reg, sentence)) for neg_reg in negative_regex]

    #search for positive regex strings in our sentence
    if positive_regex is not None:
        if isinstance(positive_regex, str):
            positive_matches.append(re.search(positive_regex, sentence))
        else:
            positive_matches = [bool(re.search(pos_reg, sentence)) for pos_reg in positive_regex]

    #if there were negative matches and no positive matches, then return False
    #if there were positive matches but no negative matches, then return True
    #if there were both positive and negative matches, then return np.nan
    #if there were no matches, then return np.nan
    if np.any(negative_matches) and not np.any(positive_matches):
        return False
    elif np.any(positive_matches) and not np.any(negative_matches):
        return True
    else:
        return np.nan
        
def identify_epilepsy_features(txt):
    """Get epilepsy features from the section in the note template"""
    #attempt split the section into its features
    feature_section_regex = r"(?im)(?:febrile |family )?\b[a-zA-Z]+,? \b[a-zA-Z/]+: {0,3}[a-zA-Z{]+"
    feature_sections = re.findall(feature_section_regex, txt, overlapped=True)
    feature_values = {"feature_text":txt, "status_epilepticus":np.nan, "febrile_history":np.nan, "intellectual_disability":np.nan, "family_history":np.nan}
    
    #regex to find the sections of interest
    status_exist_regex = r"(?im)status epilepti"
    febrile_exist_regex = r"(?im)febrile seizure"
    disability_exist_regex = r"(?im)\b(Intellectual|mental|cognitive) (delay|impairment|retardation|deficit|disabili(ty|ties))s?"
    history_exist_regex = r"(?im)family history"

    #for each section, try to see if it's explicitly written if the patient has the featurt
    febrile_values = []
    for section in feature_sections:
        
        #if it's status epilepticus, then process the value. Pick only the first instance, as the later ones may be copy forwarded
        if re.search(status_exist_regex, section) and pd.isnull(feature_values['status_epilepticus']):
            feature_values['status_epilepticus'] = get_feature_from_explicit_answer(section)
            
            
        #otherwise, check if it is febrile seizures
        elif re.search(febrile_exist_regex, section) and pd.isnull(feature_values['febrile_history']):
            febrile_values.append(get_feature_from_explicit_answer(section))
            
        #check if it's talking about intellectual disability
        elif re.search(disability_exist_regex, section) and pd.isnull(feature_values['intellectual_disability']):
            feature_values['intellectual_disability'] = get_feature_from_explicit_answer(section)
            
        #or if it's talking about family history
        elif re.search(history_exist_regex, section) and pd.isnull(feature_values['family_history']):
            feature_values['family_history'] = get_feature_from_explicit_answer(section)
        else:
            continue
    
    #once all sections have been processed, we check what the final febrile is
    febrile_idk_flag = -5
    if all(v < 0 for v in febrile_values):
        feature_values['febrile_history'] = febrile_idk_flag
    else:
        feature_values['febrile_history'] = True in febrile_values

    #for all nan values, we now perform verbose matching
    #first, set up the regex for verbose no features
    simple_verbose_no_regex = r"no[^.:]+"
    status_simple_verbose_no_regex = rf"(?im){simple_verbose_no_regex}status( epilepticus)?"
    febrile_simple_verbose_no_regex = rf"(?im){simple_verbose_no_regex}febrile seizures?"
    disability_simple_verbose_no_regex = rf"(?im){simple_verbose_no_regex}\b(Intellectual|mental|cognitive) (delay|impairment|retardation|deficit|disabili(ty|ties))s?"
    family_simple_verbose_no_regex = rf"(?im)({simple_verbose_no_regex}family history of ((seizures?)|(epilepsy)))"
    
    #construct the verbose yes family regex by parts
    family_unit_modifier_1 = r"((maternal|paternal) )"
    family_unit_modifier_2 = r"((first|second|third|fourth|fifth) )"
    family_unit_modifier_3 = r"(half( |-))"
    family_unit_members = r"((brother|sister|cousin|aunt|uncle|grandmother|grandfather|grandparent|mother|father|parent|nephew|niece|daughter|son|parent)s?)"
    family_unit_base = rf"({family_unit_modifier_1}?{family_unit_modifier_2}?{family_unit_modifier_3}?{family_unit_members})"
    family_unit_many = rf"(({family_unit_base},? )+(and|or) {family_unit_base})"
    possessor_regex = r"((' side)|('s side))"
    auxiliary_regex = r"( ?who)?( ?have (had )?| ?with | ?has (had )?| ?had )"
    history_regex = r"((a )?histor(y|ies) of )"
    epilepsies_regex = r"(seizure|absence|convulsion|epilepsy)"
    family_verbose_yes_regex_1 = rf"(?im)({family_unit_many}|{family_unit_base}){possessor_regex}?( both)?{auxiliary_regex}{history_regex}?(a )?{epilepsies_regex}"
    family_verbose_yes_regex_2 = rf"(?im){epilepsies_regex}s?( in)?( (his|her|their))? {family_unit_base}"

    #split the original text into sentences
    feature_sentences = txt.split(". ")

    #go through all sentences and see if they match any of the verbose regex patterns
    #NOTE: under this schema, once a sentence has been found that matches a particular pattern, future sentences will no longer be searched for that pattern
        #this means that if two sentences contain conflicting information for the same feature, only the first will be searched. 
        #this attempts to combat copy forwarding, under the assumption that the earlier senternce is the more recent/relevant one. 
    for sentence in feature_sentences:
        if pd.isnull(feature_values['status_epilepticus']):
            feature_values['status_epilepticus'] = get_feature_from_verbose_answer(sentence, negative_regex = status_simple_verbose_no_regex)
        if pd.isnull(feature_values['febrile_history']) or feature_values['febrile_history'] == febrile_idk_flag:
            feature_values['febrile_history'] = get_feature_from_verbose_answer(sentence, negative_regex = febrile_simple_verbose_no_regex)
        if pd.isnull(feature_values['intellectual_disability']):
            feature_values['intellectual_disability'] = get_feature_from_verbose_answer(sentence, negative_regex = disability_simple_verbose_no_regex)
        if pd.isnull(feature_values['family_history']):
            feature_values['family_history'] = get_feature_from_verbose_answer(sentence, negative_regex = family_simple_verbose_no_regex, positive_regex = [family_verbose_yes_regex_1, family_verbose_yes_regex_2])

    #if there are any other np.nan values still, and their respective feature is still mentioned in the text, assume it is true
    #we do this because there are more ways to says "yes, the patient has had status", than the semi-standardized ways to say no, they did not
    status_exist_regex = r"(?im)status epilepticus"
    febrile_exist_regex = r"(?im)febrile seizure"
    disability_exist_regex = r"(?im)\b(Intellectual|mental|cognitive) (delay|impairment|retardation|deficit|disabili(ty|ties))s?"
    history_exist_regex = r"(?im)family history"
    if pd.isnull(feature_values['status_epilepticus']) and re.search(status_exist_regex, txt):
        feature_values['status_epilepticus'] = True
    if pd.isnull(feature_values['febrile_history']) and re.search(febrile_exist_regex, txt):
        feature_values['febrile_history'] = True
    if pd.isnull(feature_values['intellectual_disability']) and re.search(disability_exist_regex, txt):
        feature_values['intellectual_disability'] = True
    if pd.isnull(feature_values['family_history']) and re.search(history_exist_regex, txt):
        #there is a very rare edge case where the features text will end on "family history Sz/epilepsy:" or "family history Sz/epilepsy: " with no other text. 
            #This is not captured by any of the rules above. It is not captured by rule codes -1 through -5, as it does not fit the : {0,3}\b[a-zA-Z]+ pattern to enter that function
            #So, do a quick search for this case and label it as np.nan if it exists
        if re.search(r"(?im)family history sz/epilepsy: {0,3}", txt):
            feature_values['family_history'] = -100
        else:
            feature_values['family_history'] = True

    #as a final step, replace all values < 0 with np.nan, as we don't need debug codes in actual use
    feature_values = {k:np.nan if cast_to_int(feature_values[k]) < 0 else feature_values[k] for k in feature_values}
    
    return feature_values

def identify_psychiatric_comorbidities(txt, psych_start_regex, return_global_only):
    """Identify what psychiatric comorbidities a patient has in the relevant note section"""
    #attempt to split section into its features
    psych_section_regex = r"(?im)(?:depression|anxiety|psychosis): {0,3}[a-zA-Z{]+"
    psych_subsections_first_words_regex = r"(?im)depression|anxiety|psychosis"
    psych_sections = re.findall(psych_section_regex, txt, overlapped=True)
    psych_values = {"psych_text":txt, "depression":np.nan, "anxiety":np.nan, "psychosis":np.nan, "has_psy_com":np.nan}
    
    #regex to find sections of interest
    depression_regex = r"(?im)depression"
    anxiety_regex = r"(?im)anxiety"
    psychosis_regex = r"(?im)psychosis"
    
    #for each section, try to see if it's explicitly written if the patient has the feature
    for section in psych_sections:
        #check for depression
        if re.search(depression_regex, section) and pd.isnull(psych_values['depression']):
            psych_values['depression'] = get_feature_from_explicit_answer(section, section_header_first_word_regex=psych_subsections_first_words_regex)
        #check for anxiety
        elif re.search(anxiety_regex, section) and pd.isnull(psych_values['anxiety']):
            psych_values['anxiety'] = get_feature_from_explicit_answer(section, section_header_first_word_regex=psych_subsections_first_words_regex)
        #check for psychosis
        elif re.search(psychosis_regex, section) and pd.isnull(psych_values['psychosis']):
            psych_values['psychosis'] = get_feature_from_explicit_answer(section, section_header_first_word_regex=psych_subsections_first_words_regex)
        else:
            continue
    
    #for all nan values, we now perform verbose matching
    #first, set up the regex for verbose no features
    simple_verbose_no_regex = r"no[^.:]+"
    depression_simple_verbose_no_regex = rf"(?im){simple_verbose_no_regex}depression"
    anxiety_simple_verbose_no_regex = rf"(?im){simple_verbose_no_regex}anxiety"
    psychosis_simple_verbose_no_regex = rf"(?im){simple_verbose_no_regex}psychosis"
    
    #split the original text into sentences
    psych_sentences = txt.split(". ")
    
    #go through all sentences and see if they match any of the verbose regex patterns
    #NOTE: under this schema, once a sentence has been found that matches a particular pattern, future sentences will no longer be searched for that pattern
        #this means that if two sentences contain conflicting information for the same feature, only the first will be searched. 
        #this attempts to combat copy forwarding, under the assumption that the earlier senternce is the more recent/relevant one. 
    for sentence in psych_sentences:
        if pd.isnull(psych_values['depression']):
            psych_values['depression'] = get_feature_from_verbose_answer(sentence, negative_regex = depression_simple_verbose_no_regex)
        if pd.isnull(psych_values['anxiety']):
            psych_values['anxiety'] = get_feature_from_verbose_answer(sentence, negative_regex = anxiety_simple_verbose_no_regex)
        if pd.isnull(psych_values['psychosis']):
            psych_values['psychosis'] = get_feature_from_verbose_answer(sentence, negative_regex = psychosis_simple_verbose_no_regex)
    
    #if there are any other np.nan values still, and their respective feature is still mentioned in the text, assume it is true
    #we do this because there are more ways to says "yes, the patient has had status", than the semi-standardized ways to say no, they did not
    if pd.isnull(psych_values['depression']) and re.search(depression_regex, txt):
        psych_values['depression'] = True
    if pd.isnull(psych_values['anxiety']) and re.search(anxiety_regex, txt):
        psych_values['anxiety'] = True
    if pd.isnull(psych_values['psychosis']) and re.search(psychosis_regex, txt):
        psych_values['psychosis'] = True
    
    #resolve the global prediction
    individual_psych_vals = np.array([psych_values['depression'], psych_values['anxiety'], psych_values['psychosis']])
    if True in individual_psych_vals:
        psych_values['has_psy_com']=True
    elif np.all(individual_psych_vals == False):
        psych_values['has_psy_com']=False
    elif np.all(pd.isnull(individual_psych_vals)):
        #if everything is still null, then check for a global yes, no, or unknown value
        flagged_txt = re.sub(psych_start_regex, "_FLAG_:", txt)
        psych_values['has_psy_com'] = get_feature_from_explicit_answer(flagged_txt, section_header_first_word_regex=psych_subsections_first_words_regex)

    if return_global_only:
        return {'psych_text':psych_values['psych_text'], 'has_psy_com':psych_values['has_psy_com']}
    else:
        return psych_values

def get_epilepsy_features(all_agg_pats, epi_features_start_regex, epi_features_end_regex):
    """Get all epilepsy features for all patients"""
    all_pat_epilepsy_features = []
    for pat_idx in range(len(all_agg_pats)):
        pat = all_agg_pats[pat_idx]
    
        for vis in pat.aggregate_visits:
            features_txt = get_note_section(vis.full_text, epi_features_start_regex, epi_features_end_regex)
            features = identify_epilepsy_features(features_txt)
    
            if features['feature_text'] != '':
                features['MRN'] = pat.pat_id
                features['visit_date'] = vis.visit_date
                all_pat_epilepsy_features.append(features)
    
    return pd.DataFrame(all_pat_epilepsy_features)

def get_seizure_types(all_agg_pats, semiology_start_regex, semiology_end_regex):
    """Get all seizure types for all patients"""
    all_pat_seizure_types = []
    for pat_idx in range(len(all_agg_pats)):
        pat = all_agg_pats[pat_idx]
    
        for vis in pat.aggregate_visits:
            semiology_txt = get_note_section(vis.full_text, semiology_start_regex, semiology_end_regex)
            seizure_types, all_descriptions_processed = identify_seizures_in_semiology(semiology_txt, semiology_start_regex)
    
            if seizure_types['semiology_text'] != '':
                seizure_types['MRN'] = pat.pat_id
                seizure_types['visit_date'] = vis.visit_date
                all_pat_seizure_types.append(seizure_types)
    
    return pd.DataFrame(all_pat_seizure_types)

def get_psych_comorbidities(all_agg_pats, psych_start_regex, psych_end_regex, return_global_only=True):
    """Get all psychiatric comorbidities for all patients"""
    all_pat_psych_comorbidities = []
    for pat_idx in range(len(all_agg_pats)):
        pat = all_agg_pats[pat_idx]

        for vis in pat.aggregate_visits:
            psych_txt = get_note_section(vis.full_text, psych_start_regex, psych_end_regex)
            psych_comorbidities = identify_psychiatric_comorbidities(psych_txt, psych_start_regex, return_global_only)

            if psych_comorbidities['psych_text'] != '':
                psych_comorbidities['MRN'] = pat.pat_id
                psych_comorbidities['visit_date'] = vis.visit_date
                all_pat_psych_comorbidities.append(psych_comorbidities)

    return pd.DataFrame(all_pat_psych_comorbidities)
    
def get_cohort_confounders(all_agg_pats, epi_features_regex, semiology_regex, psych_comorbidities_regex):
    """Get covariates for the cohort"""
    return get_epilepsy_features(all_agg_pats, epi_features_regex['start'], epi_features_regex['end']), get_seizure_types(all_agg_pats, semiology_regex['start'], semiology_regex['end']), get_psych_comorbidities(all_agg_pats, psych_comorbidities_regex['start'], psych_comorbidities_regex['end'])