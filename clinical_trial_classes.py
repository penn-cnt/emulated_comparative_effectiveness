import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import clinical_trial_functions as ctf
import data_loading_functions as dlf
import sys
import pickle
import pipeline_utilities as pu
from copy import deepcopy


class PrescriptionRecord():
    """
    Contains information on a set of prescriptions for a single ASM
        patient: the patient who has this set of prescription
        name: the name of the medication
        order_date: when the prescription was ordered
        start_date: the earliest start date among all prescriptions for this ASM
        end_date: the latest end date among all prescriptions for this ASM
        prescriptions: a set containing all prescriptions for this single ASM
    """
    def __init__(self, patient, name, order_date, start_date, end_date):
        self.patient = patient
        self.name = name
        self.end_date = end_date
        self.prescriptions = set()
        
        #check if this prescription record has a valid start date, past 1970
        if start_date <= datetime(year=1970, month=1, day=1):
            start_date = np.nan
            print(f"Invalid start_date in PrescriptionRecord for patient {patient.pat_id}")
        
        if not pd.isnull(start_date):
            self.start_date = start_date
        else:
            self.start_date = order_date
        
    def add_Prescription(self, new_Prescription):
        """Adds another Prescription to this PrescriptionRecord"""
        self.prescriptions.add(new_Prescription)
    
    def update_PrescriptionRecord(self, new_PrescriptionRecord):
        """Updates the existing PrescriptionRecord if a new one is introduced"""
        #Prescriptions are equal only on patient and name. 
        if self != new_PrescriptionRecord: 
            raise TypeError
        
        #update the end date
        #always update if the current value is null
        if pd.isnull(self.end_date):
            self.end_date = new_PrescriptionRecord.end_date
        else:
            self.end_date = new_PrescriptionRecord.end_date if new_PrescriptionRecord.end_date > self.end_date else self.end_date
        
        #update the start date
        #always update if the current value is null
        if pd.isnull(self.start_date):
            self.start_date = new_PrescriptionRecord.start_date
        else:
            self.start_date = new_PrescriptionRecord.start_date if new_PrescriptionRecord.start_date < self.start_date else self.start_date
            
        self.prescriptions = self.prescriptions.union(new_PrescriptionRecord.prescriptions)
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.patient == self.patient and other.name == self.name
        else:
            return False
        
    def info(self):
        print(f"{self.patient.pat_id}, {self.name}, {self.start_date}, {self.end_date}")
        
class Prescription():
    """
    Contains information from a single prescription of an ASM
        patient: the patient who is prescribed the medication
        name: the name of the ASM
        description: the description of the prescription, as seen in the raw data
        order_date: when the prescription was ordered
        start_date: when the prescription begins
        end_Date: when the prescription ends
        dose: the dose of the prescription
        dose_unit: the unit of the dose
        frequency: how often the patient must take the ASM
    """
    def __init__(self, patient, name, description, order_date, start_date, end_date, dose, dose_unit, frequency):
        self.patient = patient
        self.name = name
        self.description = description
        self.end_date = end_date
        self.frequency = frequency
        
        #if we have nan doses and dose_units, or non-standard (ml, mg) dose units, then infer from description
        if pd.isnull(dose) or pd.isnull(dose_unit) or (dose_unit.lower() != 'ml') or (dose_unit.lower() != "mg"):
            self.dose, self.dose_unit = self.__infer_dose__(self.description, dose, dose_unit)
        #otherwise, use the actual doses
        else:
            self.dose = dose
            self.dose_unit = dose_unit
            
        #if the dose is over 10,000, something has to be very wrong in the record. Set these to NaNs
        if not isinstance(self.dose, str):
            if self.dose >= 10000:
                self.dose = np.nan
                
        #check if this prescription has a valid start date, past 1970
        if start_date <= datetime(year=1970, month=1, day=1):
            start_date = np.nan
            print(f"Invalid start_date in Prescription for patient {patient.pat_id}")
            
        #try and get a complete start date
        if not pd.isnull(start_date):
            self.start_date = start_date
        else:
            self.start_date = order_date
            
        #pre-compute the hash, as this class should be immutable
        self._hash = hash((self.patient.pat_id, self.name, self.description, self.start_date, self.end_date, self.dose, self.dose_unit, self.frequency))
            
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.patient == self.patient and other.name == self.name and self.dose == other.dose and self.dose_unit == other.dose_unit and self.start_date == other.start_date and self.end_date == other.end_date and self.frequency == other.frequency
        else:
            return False
        
    def __hash__(self):
        return self._hash

    def info(self):
        return f"{self.patient.pat_id}, {self.name}, {self.dose} {self.dose_unit}, {self.frequency}, {self.start_date}, {self.end_date}"
    
    def __infer_dose__(self, description, original_dose, original_dose_unit):
        """Attempts to calculate the dose of the prescription from its description"""
        #split the description into words
        desc_split = description.lower().split()
        
        #for each word, see if it's a float. 
        is_dose = np.array([dlf.str_to_float(word) for word in desc_split])
        dose_idx = np.argwhere(is_dose > 0).flatten()
        
        #for each word, see if it's a common unit (ml, mg)
        is_unit = [(word == 'ml' or word == 'mg') for word in desc_split]
        unit_idx = np.argwhere(is_unit).flatten()
        
        likely_dose = np.nan
        likely_unit = None
        #indices where is_dose == true immediately followed by is_unit == True is almost certainly the value and unit
        for idx in dose_idx:
            if (idx+1) in unit_idx:
                likely_dose = dlf.str_to_float(desc_split[idx])
                likely_unit = desc_split[idx+1]
                
        #check if our likely values are nan. If they are, return the original values
        if pd.isnull(likely_dose) or pd.isnull(likely_unit):
            return original_dose, original_dose_unit
        
        #check if we have nan original dose and/or units. If so, return the description values
        if pd.isnull(original_dose) or pd.isnull(original_dose_unit):
            return likely_dose, likely_unit
                
        #check if the dose unit is in tablets, capsules, etc...
        #if it is, multiple the number of tabs by the likely_dose
        if "tab" in original_dose_unit.lower() or "cap" in original_dose_unit.lower() or "each" in original_dose_unit.lower():
            #first, check that the original dose doesn't equal the likely dose or is a multiple of the likely dose. 
            #If it does, then it's probably a typo in the record, where the unit should not have been capsule or tablet. Return the original dose, likely_unit
            if np.any([np.abs(original_dose - (likely_dose * i)) < 1 for i in range(5)]):
                return original_dose, likely_unit
            else:
                return likely_dose * original_dose, likely_unit
        
        return original_dose, original_dose_unit
    
class TrialSubject():
    """
    Contains information for a single patient in a clinical trial
        aggregate_patient: the AggregatePatient object that contains their general information and outcomes data
        clinicalTrial: the clinical trial that the patient is being enrolled in
        prior_asms: the ASMs the patient has taken prior to the clinical trial (as a set)
        pre_trial_visits: the visits the patient had prior to the clinical trial
        trial_visits: the visits the patient had during the clinical trial
        post_trial_visits: the visits the patient had after the clinical trial
        trial_Prescriptions: what Prescriptions did the patient have during this trial?
        confounders: what covariates does the patient have for propensity scoring?
    """
    def __init__(self, aggregate_patient, clinicalTrial):
        #which patient is this TrialSubject referencing?
        self.patient = aggregate_patient
        #which trial is the subject potentially enrolled in?
        self.clinicalTrial = clinicalTrial

        #what ASMs were they taking prior to starting the current clinicalTrial?
        self.prior_asms = set()
        self.num_prior_asms = np.nan
        
        #sort the visits by visit date
        self._sorted_visits = np.array(sorted(self.patient.aggregate_visits, key=lambda x:x.visit_date))
        self._sorted_visit_dates = np.array([vis.visit_date for vis in self._sorted_visits])
        
        #which visits for this patient fall within the trial's parameters?
        self.pre_trial_visits = []
        self.trial_visits = []
        self.post_trial_visits = []
        #which Prescriptions for this patient fall within the trial's parameters?
        self.trial_Prescriptions = []
        
        #how much total time is this patient in the trial?
        self.time_in_trial = timedelta(days=-99999)
        self.time_in_pre_trial = timedelta(days=-99999)
        #why did the patient leave the trial?
        self.trial_exit_reason = None

        #what is their propensity score information
        self.confounders = None
        self.propensity_score = {'overall':None}

    def determine_confounders(self, confounder_table, metadata_cols = ['MRN', 'visit_date', 'DOB_YR']):
        """ Calculates the values of the covariates for this patient """
        #find all confounder entries for this patient with visit_date up to the start date of the trial
        #sort them by visit date
        trial_start = self.patient.medications[self.clinicalTrial.name].start_date
        subject_confounder_table = confounder_table.loc[(confounder_table.MRN == self.patient.pat_id) & (confounder_table.visit_date <= trial_start)].sort_values(by='visit_date')

        #check if confounders for this patient were found
        if len(subject_confounder_table) == 0:
            return -1

        # get the most relevant confounders
        epiType_encoder = {'Other':2, 'Focal':1, 'Generalized':0}
        self.confounders = {}
        self.confounders['GTCS'] = True in subject_confounder_table.GTCS.values
        self.confounders['other_Sz_Types'] = True in subject_confounder_table.other_Sz_Types.values
        self.confounders['status_epilepticus'] = True in subject_confounder_table.status_epilepticus.values
        self.confounders['febrile_history'] = True in subject_confounder_table.febrile_history.values
        self.confounders['intellectual_disability'] = True in subject_confounder_table.intellectual_disability.values
        self.confounders['family_history'] = True in subject_confounder_table.family_history.values
        self.confounders['has_psy_com'] = True in subject_confounder_table.has_psy_com.values
        self.confounders['sex'] = subject_confounder_table.GENDER.mode().iloc[0]
        self.confounders['epiType'] = epiType_encoder[self.patient.epiType]
        self.confounders['therapy_type'] = 0 if len(self.trial_Prescriptions) == 1 else 1 #0 for monotherapy, 1 for polytherapy
        self.confounders['start_year'] = trial_start.year
        
        #we want to bin the age at trial start, and number of prior asms
        #converting them from continuous variables into categorical variables
        #age bins: 18-39, 40-64, 65+, mapping to 0, 1, 2 respectively
        #ASM bins: 0, 1, 2, 3+, mapping to 0, 1, 2, 3, respectively
        age_at_trial_start = trial_start.year - subject_confounder_table.iloc[-1].DOB_YR
        if age_at_trial_start >= 18 and age_at_trial_start < 40:
            self.confounders['age_at_trial_start'] = 0
        elif age_at_trial_start >= 40 and age_at_trial_start < 65:
            self.confounders['age_at_trial_start'] = 1
        else:
            self.confounders['age_at_trial_start'] = 2
        self.confounders['num_prior_asms'] = np.min([len(self.prior_asms), 3])

        #we return 1 instead of void so we can use this in a cheeky list comprehension in our trial functions.
        if np.any(pd.isnull(list(self.confounders.values()))):
            return -2
        else:
            return 1

    def _get_most_recent_nonNaN(table, column):
        try:
            return table.dropna(subset=column).iloc[-1][column]
        except:
            return np.nan

    def _get_most_common_nonNaN(table, column):
        try:
            return table.dropna(subset=column)[column].mode().iloc[0]
        except:
            return np.nan

    def check_trial_requisites(self, medication_start_regex, medication_end_regex, brand_to_generic, 
                               require_trial_visits=True, require_pre_trial_visits=True, retention_outcome=False):
        """ check if this subjects meets the enrollment criteria """
        #some flags for debugging purposes. We default None instead of False so that when we return, we can know where we're returning from in the code
        has_pre_trial_visits = None
        has_trial_visits = None
        has_required_num_asms = None
        
        #check that this patient has the trial's medication and it is not a prior asm
        if (self.clinicalTrial.name not in self.patient.medications) or (self.clinicalTrial.name in self.patient.medications['prior_asms']):
            return has_required_num_asms, has_trial_visits, has_pre_trial_visits

        #if the trial uses retention as its outcome measure, then the enrollment end date is when their prescription ends
        if retention_outcome:
            self.trial_exit_reason = 'no_newer_prescription_for_current_ASM'
            time_in_trial = self.patient.medications[self.clinicalTrial.name].end_date - self.patient.medications[self.clinicalTrial.name].start_date
            enrollment_end_date = self.patient.medications[self.clinicalTrial.name].end_date
            
        else:
            #does the subject have a PrescriptionRecord of another prescription with a start_date after this drug?
            #calculate the minimum number of days until the next prescriptionRecord
            days_until_next_record = timedelta(days=99999)
            for med in self.patient.medications:
                if med == self.clinicalTrial.name or med == 'prior_asms':
                    continue
                days_to_record = (self.patient.medications[med].start_date - self.patient.medications[self.clinicalTrial.name].start_date)
                if days_to_record >= timedelta(days=0):
                    days_until_next_record = days_to_record if days_to_record < days_until_next_record else days_until_next_record
            
            #now, calculate the time in the trial
            time_in_trial = np.min([days_until_next_record, self.patient.medications[self.clinicalTrial.name].end_date - self.patient.medications[self.clinicalTrial.name].start_date])
            if days_until_next_record < self.patient.medications[self.clinicalTrial.name].end_date - self.patient.medications[self.clinicalTrial.name].start_date:
                self.trial_exit_reason = 'prescription_for_new_ASM'
            else:
                self.trial_exit_reason = 'no_newer_prescription_for_current_ASM'

            #when does the patient stop being in this trial?
            enrollment_end_date = self.patient.medications[self.clinicalTrial.name].start_date + time_in_trial

        #the visits during the trial occur between the start date and enrollment end date
        #exclusive on the lower bound. Visits that occur on the trial start date are part of the baseline interval
        trial_visits = [vis for vis in self._sorted_visits if (vis.visit_date > self.patient.medications[self.clinicalTrial.name].start_date) and (vis.visit_date <= enrollment_end_date)]

        #if the patient has at least one visit during the trial period, then they are elligible for the trial. Or, we don't care about trial visits (drug retention outcome)
        if ((require_trial_visits and (len(trial_visits) > 0)) or (not require_trial_visits)) and time_in_trial.days > 0:
            #get trial info
            self.trial_visits = trial_visits
            self.post_trial_visits = [vis for vis in self._sorted_visits if (vis.visit_date > enrollment_end_date)]
            self.time_in_trial = time_in_trial

            #the prescriptions during the trial occur between the start date and enrollment end date. inclusive on the lower bound, as the prescriptions are prospective
            self.trial_Prescriptions = [prescription for prescription in self.patient.medications[self.clinicalTrial.name].prescriptions if (prescription.start_date >= self.patient.medications[self.clinicalTrial.name].start_date) and (prescription.start_date < enrollment_end_date)]
            self.trial_Prescriptions = sorted(self.trial_Prescriptions, key=lambda x: x.start_date)
            has_trial_visits=True     
        else:
            has_trial_visits = False
            return has_required_num_asms, has_trial_visits, has_pre_trial_visits
        
        #get the visits before the trial
        pre_trial_visits = [vis for vis in self._sorted_visits if vis.visit_date <= self.patient.medications[self.clinicalTrial.name].start_date]

        #if the patient has at least one visit before the trial period, then they are elligible for the trial. Or, we don't care about the pre-trial visits (drug retention outcome)
        if (require_pre_trial_visits and (len(pre_trial_visits) > 0)) or (not require_pre_trial_visits):
            self.pre_trial_visits = pre_trial_visits
            self.time_in_pre_trial = self.patient.medications[self.clinicalTrial.name].start_date - self.pre_trial_visits[0].visit_date if (len(pre_trial_visits) > 0) else np.nan
            has_pre_trial_visits = True
        else:
            has_pre_trial_visits = False
            return has_required_num_asms, has_trial_visits, has_pre_trial_visits

        #find the prior ASMs
        self.prior_asms = deepcopy(self.patient.medications['prior_asms'])
        #what about new prescriptions before the current one?
        for med in self.patient.medications:
            if med == self.clinicalTrial.name or med == 'prior_asms':
                continue
            if self.patient.medications[med].start_date < self.patient.medications[self.clinicalTrial.name].start_date:
                self.prior_asms.add(med)
        #scan through all pre-trial-visits for additional asms
        for vis in self.pre_trial_visits:
            prior_asm_txt = dlf.get_note_section(vis.full_text, medication_start_regex, medication_end_regex)
            pre_trial_prior_asms = dlf.get_prior_ASMs_from_note(prior_asm_txt, brand_to_generic)
            self.prior_asms = self.prior_asms.union(pre_trial_prior_asms)
        
        #calculate the number of prior ASMs
        self.num_prior_asms = len(self.prior_asms)

        #check that the patient falls within the minimum and maximum number of prior ASMs allowed in the study
        if not pd.isnull(self.clinicalTrial.minimum_prior_asms):
            if self.num_prior_asms < self.clinicalTrial.minimum_prior_asms:
                has_required_num_asms = False
                return has_required_num_asms, has_trial_visits, has_pre_trial_visits 
        if not pd.isnull(self.clinicalTrial.maximum_prior_asms):
            if self.num_prior_asms >= self.clinicalTrial.maximum_prior_asms:
                has_required_num_asms = False
                return has_required_num_asms, has_trial_visits, has_pre_trial_visits 
        has_required_num_asms = True
            
        return has_required_num_asms, has_trial_visits, has_pre_trial_visits
        
            
        
class ClinicalTrial():
    """
    Contains information for clinical trials
    """
    def __init__(self, medication_name, outcome_measure, minimum_prior_asms=None, maximum_prior_asms=None):
        self.name = medication_name #the name of the ASM being considered
        self.minimum_prior_asms = minimum_prior_asms #patient must have taken at least this number of prior ASMs (inclusive)
        self.maximum_prior_asms = maximum_prior_asms #patient must have taken less than this number of prior ASMs (exclusive)

        if outcome_measure.lower() not in ['sustained_freedom', 'freedom', 'frequency', 'retention']:
            raise ValueError(f"Error: Unrecognized outcome measure. You asked for {outcome_measure}. ")
        else:
            self.outcome_measure = outcome_measure
        
        self.cohort = [] #A dictionary of trial outcomes, each with their own cohort
        self._hash = hash((self.name, self.minimum_prior_asms, self.maximum_prior_asms))
        
    def enroll_subjects(self, agg_pats, medication_start_regex, medication_end_regex, brand_to_generic, debug=False):
        """ add subjects to the clinical trial from a list of agg_pats """

        #some counters for debugging
        pre_trial_vis_fail = 0
        trial_vis_fail = 0
        initial_fail = 0
        num_asms_fail = 0

        #iterate through the patients
        for pat in agg_pats:
            
            #create a TrialSubject for the patient
            subject = TrialSubject(pat, self)
            
            #check and see if this subject has the necessary trial requisites
            if self.outcome_measure == 'retention':
                requisites_matched = subject.check_trial_requisites(medication_start_regex, medication_end_regex, brand_to_generic, 
                                                                    require_trial_visits=False, require_pre_trial_visits=False, retention_outcome=True)
                
                if np.all(requisites_matched):
                    self.cohort.append(subject)
                elif debug:
                    if requisites_matched == (None, None, None):
                        initial_fail += 1
                    elif requisites_matched[0] == False:
                        num_asms_fail += 1
                    elif requisites_matched[1] == False:
                        trial_vis_fail += 1
                    elif requisites_matched[2] == False:
                        pre_trial_vis_fail += 1    
                    else:
                        return pat
            else: #seizure frequency trial
                requisites_matched = subject.check_trial_requisites(medication_start_regex, medication_end_regex, brand_to_generic,
                                                                    require_trial_visits=True, require_pre_trial_visits=True, retention_outcome=False)
                
                if np.all(requisites_matched):
                    self.cohort.append(subject)
                elif debug:
                    if requisites_matched == (None, None, None):
                        initial_fail += 1
                    elif requisites_matched[0] == False:
                        num_asms_fail += 1
                    elif requisites_matched[1] == False:
                        trial_vis_fail += 1
                    elif requisites_matched[2] == False:
                        pre_trial_vis_fail += 1
                    else:
                        return pat

        if debug:
            print(f"Initial Fails: {initial_fail}\nWrong number of prior ASMs: {num_asms_fail}\nTrial visit failure: {trial_vis_fail}\nPre-trial visit failure: {pre_trial_vis_fail}")
        
    def __hash__(self):
        return self._hash