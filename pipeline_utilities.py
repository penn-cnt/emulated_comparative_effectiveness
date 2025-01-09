from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import regex as re
import string
import json
import os
import sys

class run_silently():
    """A helper function to disable print statements. 
    Copied from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def export_HF_dataset_to_json(dataset, path):
    """ Export a huggingface formatted dataset into a json """
    with open(path, 'w') as f:
      for datum in dataset:
        json.dump(datum, f)
        f.write('\n')

def replace_implicits(text, implicit_converter):
    """
    Replaces words like "many" and "some" with the desired numerical conversion
    Input: 
        text: The text to process
        implicit_converter: The conversion dictionary
    Returns:
        The converted text
    """
    for imp in implicit_converter:
            text = text.replace(str(imp), str(implicit_converter[imp]))
    return text

def load_section_regex():
    """Load regex patterns needed to parse notes for specific sections"""
    time_pattern = r"((past|prior|current|accidental|active|previous) )"
    AS_pattern = r"anti( |-)?(seizure|epileptic) "
    base_med_pattern = r"(prescription|medication|drug)s?"
    base_asm_pattern = r"(asm|asd|aed)s?"
    ASM_pattern = rf"({time_pattern}{AS_pattern}{base_med_pattern})|({time_pattern}{base_asm_pattern})|({base_asm_pattern} tried)|(outpatient prescriptions marked as taking)"
    medication_pattern = rf"{time_pattern}(Outpatient )?{base_med_pattern}"
    
    sz_pattern = r"((seizure|event)|(seizure/event)) "
    seizure_desc_pattern = rf"{sz_pattern}descriptions?"
    semiology_pattern = rf"({sz_pattern})?(semiology|semiologies)"
    features_section_pattern = r"(Special Features?)|(\bEpilepsy (Risk )?Factors?)"
    semiology_section_pattern = rf"({semiology_pattern})|({seizure_desc_pattern})|(typical events)"
    
    type_pattern = r"(Social|Surgical|Medical|Psychiatric|Seizure|Disease|Epilepsy|Documented|OB/GYN)"
    history_pattern = rf"{time_pattern}?{type_pattern} (history|hx)"
    study_pattern = rf"{time_pattern}({type_pattern} )?studies"
    
    exam_pattern = r"(\bLab)|(\bExam)|(\bDiagnostic\b)|(\bPE\b)"
    plan_pattern = r"(\bImpression)|(\bPlan\b)"
    hpi_pattern = r"(\bHistory of Present Illness\b)|(\bInterval History\b)|(\bHPI\b)"
    other_pattern = r"(\bAllerg)|(\bIssues\b)|(\bMedical Problems?)|(Chief complaint)"

    return time_pattern, AS_pattern, base_med_pattern, base_asm_pattern, ASM_pattern, medication_pattern,\
        sz_pattern, seizure_desc_pattern, semiology_pattern, features_section_pattern, semiology_section_pattern,\
        type_pattern, history_pattern, study_pattern, exam_pattern, plan_pattern, hpi_pattern, other_pattern

def keep_longest_note(df, text_col, sort_subset_cols):
    """ 
    Finds the longest note in a dataframe of notes for each patient. Notes are matched based on sort_subset_cols of identifiers
    Input:
        df: the dataframe of notes
        text_col: the column in the dataframe with the note text
        sort_subset_cols: the columns in the dataframe that contain patient identifiers for each note
    Returns:
        a dataframe with only the longest note for each patient
    """
    df['note_len'] = df.apply(lambda x: len(str(x[text_col])), axis=1)
    return df.sort_values(by='note_len').drop_duplicates(subset=sort_subset_cols, keep='last').drop('note_len', axis=1)

def find_attending_addendums(x, starting_addendum_string, mid_addendum_string):
    """
        Identifies if a note is an attending addendum
    """
    #first pass: check if addend is early in the note
    if "addend" in x.lower()[:100]:
        return True
    #2nd pass: check if the initial text contains the starting_addendum_string
    if SequenceMatcher(None, x[:int(len(starting_addendum_string)*1.5)], starting_addendum_string).ratio() > 0.66:
        return True
    #3rd pass: check if the first 10 sentences have mid_addendum_string
    return np.any([SequenceMatcher(None, sentence, mid_addendum_string).ratio() > 0.75 for sentence in x.split(". ")[:10]])

def flatten_python_list(l):
    """
        Flattens a python list of lists.
        from: https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    """    
    return [item for sublist in l for item in sublist]

def get_section_from_note(txt, start_regex, end_regex, splitter="    "):
    """
    Finds a section from a note between start_regex and end_regex
    Input:
        txt: The note text
        start_regex: the regex pattern to find the start of the section
        end_regex: the regex pattern to find the end of the section (or the start of other sections)
    Returns:
        the text of the section
    """
    txt = txt.replace(u'\xa0', u' ')
    lines = txt.split(splitter)

    #get all possible starting points
    prior_start_idx = [idx for idx in range(len(lines)) if re.search(start_regex, lines[idx][:75])]
    all_end_idx = [idx for idx in range(len(lines)) if re.search(end_regex, lines[idx][:75])]
    
    #The earliest ending point right after the earliest start point
    section_text = []
    for j in range(len(prior_start_idx)):
        earliest_start_idx = prior_start_idx[j]
        possible_end_idx = np.array(all_end_idx) > earliest_start_idx
        if len(possible_end_idx) > 0:
            earliest_end_idx = all_end_idx[np.argmax(possible_end_idx)]
        else:
            continue
        
        #check to make sure the earliest end idx > earliest start idx
        if earliest_end_idx <= earliest_start_idx:
            continue
        
        #stitch the strings together
        section_text.append(" ".join(lines[earliest_start_idx:earliest_end_idx]).strip())
    
    return section_text

def get_paragraph_with_max_token_length(tokenizer, start_regex, end_regex, note_text, note_author, pat_id, visit_date, note_id, enc_id, splitter="  ", max_token_length=512, fill_if_empty=False, debug=False):
    """
    Load a paragraph of text given a medical note. Truncates at maximum token length
    Input:
        tokenizer: A Hugging Face Tokenizer to use to calculate length of tokens
        start_regex: the regex pattern to find the start of the section
        end_regex: the regex pattern to find the end of the section
        note_text: The note's text
        note_author: The note's author
        pat_id: The patient's ID
        visit_date: The date of the visit
        note_id: The note's identifier
        enc_id: The encounter's identifier
        splitter: How are new lines in the note text split from each other?
        max_token_length: How many tokens to keep for truncation
        fill_if_empty: should we use the first max_token_length tokens of the document if no sections are found?
    Returns: a dictionary containing the document's info
    """
    #get the note sections
    #check if start_regex is a string or list of strings
    if isinstance(start_regex, str) and isinstance(end_regex, str):
        section_texts = get_section_from_note(note_text.strip('"'), start_regex, end_regex, splitter)
    elif isinstance(start_regex, list) and isinstance(end_regex, list):
        if len(start_regex) != len(end_regex):
            raise ValueError("Error - starting and ending regex don't have the same length")
        section_texts = []
        for i in range(len(start_regex)):
            section_texts += get_section_from_note(note_text.strip('"'), start_regex[i], end_regex[i], splitter)

    #Dictionary to store relevant information of the document
    document = {}
    document['filename'] = f"{pat_id}_{note_id}_{enc_id}_{note_author}_{visit_date[:10]}"
    document['note_author'] = note_author
    extract_counter = 0

    if len(section_texts) >= 1:
        #truncate them to max_length
        section_texts = [tokenizer.decode(tokenizer(txt.strip(), max_length=max_token_length, stride=128, truncation='do_not_truncate', add_special_tokens=False)['input_ids']) for txt in section_texts]

        #add them into the document
        for txt in section_texts:
            #if there are at least 100 characters in the text, keep it in consideration
            if len(txt) < 100:
                continue

            #find where it says "previous/prior history" and truncate there
            prior_history_regex = r"(?im)(past|prior|previous|initial) (history|hx):"
            document[extract_counter] = re.split(prior_history_regex, txt)[0]
            extract_counter += 1
        
    #if no paragraphs were found, then take the first max_token_length tokens of the text.
    elif len(document) <= 2 and fill_if_empty:        
        #truncate to max_token_length
        doc_text = tokenizer.decode(tokenizer(note_text, max_length=max_token_length, stride=128, truncation='do_not_truncate', add_special_tokens=False)['input_ids'])
        document[extract_counter] = doc_text
        extract_counter += 1
    
    if debug:
        return [document, section_texts, start_regex, end_regex]
    else:
        return document
    
def translate_summaries(predictions_and_summaries):
    """
    Converts the summaries generated by the T5 model into numbers or dates. Modifies, in place, the input pd.DataFrame
    Input:
        predictions_and_summaries: a pd.DataFrame with columns:
            prediction: the finetuned RoBERTa predictions to extract seizure frequency and date of last occurrence from note text
            summarization: the summarization of the prediction provided by the T5 model
            id: the (question/answer) id of the prediction and summarization
            'sz_per_month': the column to store the seizure frequency translations, as a float in units of per month
            'last_occurrence': the column to store the last seizure occurrence translation, as a datetime.            
    """
    for idx in predictions_and_summaries.index:        
        try:
            #remove commas inside the summary (commas mess up formatting)
            predictions_and_summaries.loc[idx, 'summarization'] = predictions_and_summaries.loc[idx, 'summarization'].translate(str.maketrans('', '', ','))
            
            #check if it is a pqf or elo summary
            if 'pqf' in predictions_and_summaries.loc[idx, 'id'].lower():
                #if the summary is blank, skip it
                if predictions_and_summaries.loc[idx, 'summarization'] == "":
                    continue

                #check if it is a seizure calendar
                szCal = process_calendar(predictions_and_summaries.loc[idx, 'prediction'])
                if szCal is not None:
                    predictions_and_summaries.loc[idx,  'sz_per_month'] = szCal

                #if it isn't a seizure calendar, then process it normally
                else: 
                    
                    #split the summary by spaces
                    summary = predictions_and_summaries.loc[idx, 'summarization'].split()

                    #check if the 0th and 2nd indices are numeric
                    if not summary[0].isnumeric(): #if it isn't a number, it's probably a range. Split it by a dash
                        summary[0] = summary[0].split('-')[-1]
                        #if it still isn't a number, it could be an "or", in which case split it by '/'
                        if not summary[0].isnumeric(): 
                            summary[0] = summary[0].split('/')[-1]
                    if not summary[2].isnumeric():
                        summary[2] = summary[2].split('-')[-1]
                        #if it still isn't a number, it could be an "or", in which case split it by '/'
                        if not summary[2].isnumeric(): 
                            summary[2] = summary[0].split('/')[-1]

                    #if it is a "per visit" frequency, run the per-visit translator
                    if 'visit' in summary[-1].lower():
                        predictions_and_summaries.loc[idx, 'sz_per_month'] = 'calc_since_last_visit_pqf()'

                    #if it is a lifetime frequency, run the lifetime translator
                    elif 'life' in summary[-1].lower():
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = 'calc_lifetime_pqf()'

                    #we want to convert everything to per month.
                    #there are 12 months/year, and therefore 0.0833 years/month
                    #there are 365 days/year, 12 months/year, and therefore 30.4167 days/month
                    #there are 365 days/year, 12 months/year, 7 days/week, and therefore 4.3452 weeks/month
                    elif 'month' in summary[-1].lower(): #if it's per month, just calculate dx/dt using the existing values
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) / float(summary[2]) 
                    elif 'day' in summary[-1].lower() or 'night' in summary[-1].lower(): #if it's per day or per night, multiple the numerator by 30.4167
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 30.4167 / float(summary[2])
                    elif 'week' in summary[-1].lower(): #it it's per week, multiply the numerator by 4.3452
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 4.3452 / float(summary[2])
                    elif 'year' in summary[-1].lower(): #if it's per year, multiply the numerator by 0.0833
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 0.0833 / float(summary[2])
                    elif 'hour' in summary[-1].lower(): #if it's per hour, multiply the numerator by 24 and then by 30.4167
                        predictions_and_summaries.loc[idx,  'sz_per_month'] = float(summary[0]) * 24 * 30.4167 / float(summary[2])
                    else:
                        print(f"ERROR - PQF timeframe unidentifiable. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
            
            #otherwise, check if it is a date of last seizure
            elif 'elo' in predictions_and_summaries.loc[idx, 'id'].lower():
                #if the summary is blank, skip it
                if predictions_and_summaries.loc[idx, 'summarization'] == "":
                    continue

                #split the summary by spaces
                summary = predictions_and_summaries.loc[idx, 'summarization'].split()

                #first, check if it is an "ago" last occurrence of the form X Y ago
                if 'ago' in summary[-1].lower():

                    #check if the 0th index is numeric
                    if not summary[0].isnumeric(): #if it isn't a number, it's probably a range. Split it by a dash and take the max
                        summary[0] = summary[0].split('-')[-1]

                        #if it still isn't a number, it could be an "or", in which case split it by '/'
                        if not summary[0].isnumeric(): 
                            summary[0] = summary[0].split('/')[-1]

                    #get the Y (timeframe) of the ago "months, days, years, etc..."
                    if 'day' in summary[1] or 'night' in summary[1]: #if it is days ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0]))
                    elif 'week' in summary[1]: #if it is weeks ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0])*7)
                    elif 'month' in summary[1]: #if it is months ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0])*30.4167)
                    elif 'year' in summary[1]: #if it is years ago, calculate the date
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = predictions_and_summaries.loc[idx,  'visit_date'] - timedelta(days=float(summary[0])*365)
                    else:
                        print(f"ERROR - ELO 'ago' timeframe unidentifiable. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                    continue

                #otherwise, it must be some sort of date.
                days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
                is_month = False

                #the number of splits determines the format of the date
                if len(summary) == 1: #if the summary was only a single word,
                    #check if it is a number, in which case it must be a year
                    if summary[0].isnumeric():
                        #check if the year has 4 digits
                        if len(summary[0]) != 4:
                            print(f"ERROR - ELO summary had 1 item that suggests a year, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                        #if it looks ok, convert it to a datetime. Python defaults the date to Jan 1st!!!!
                        summary = datetime.strptime(summary[0], '%Y')

                        #check if the year is within a reasonable range [1850-today]
                        if summary > datetime.today() or summary < datetime(year=1850, month=1, day=1):
                            print(f"ERROR - ELO summary had 1 item that suggests a year, but it was out of bounds. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                        #the year is then the last occurrence.
                        predictions_and_summaries.loc[idx,  'last_occurrence'] = summary

                    #if it isn't a number, then it is a single month or single day
                    else:
                        try: #try to convert it as a month
                            elo_date = datetime.strptime(summary[0], '%B').replace(year = predictions_and_summaries.loc[idx,'visit_date'].year)
                            #if this proposed date is > the visit date, then move the year back by 1
                            if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                elo_date = elo_date.replace(year = predictions_and_summaries.loc[idx,'visit_date'].year - 1)

                            predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date

                        except: #if you can't convert it as a month, try as a day
                            for i in range(len(days)): #check days
                                if days[i] in summary[0].lower():                         
                                    #count backwards from the visit date by day until we get to the right weekday
                                    visit_date = predictions_and_summaries.loc[idx,'visit_date']
                                    break_ct = 0
                                    while i != visit_date.weekday():
                                        visit_date -= timedelta(days=1)
                                        break_ct += 1
                                        if break_ct > 7:
                                            print(f"ERROR - date backtracking did not terminate in time. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                            break

                                    #this final date is the last occurrence
                                    predictions_and_summaries.loc[idx,  'last_occurrence'] = visit_date

                                    break

                        #if it was neither a or day, warn the user
                        if predictions_and_summaries.loc[idx,  'last_occurrence'] == -1:
                            print(f"ERROR - ELO summary had 1 item, but could translate neither a month, day, nor year. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                #if the date is given as 2 items it is either month, year; or month, day (year implicit);
                elif len(summary) == 2:

                    #check if the last item could be a year
                    if summary[-1].isnumeric() and len(summary[-1]) == 4:
                        try:
                            #if it is a year, then the first item must be a month. Otherwise, it doesn't make sense to give day, year without the month
                            if len(summary[0]) > 3:
                                elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %Y')
                            else:
                                elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%b %Y')

                            #if this proposed date is > the visit date, report an error
                            if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                print(f"ERROR - ELO summary had 2 items that suggest month year, but exceeds visit date. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                continue

                            #this final date is the last occurrence
                            predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date

                        except:
                            print(f"ERROR - ELO summary had 2 items that suggest month year, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue

                    #if it is not a year, then it must be a day. check if so
                    elif summary[-1].isnumeric() and (len(summary[-1]) <= 2 and len(summary[-1]) > 0):
                        day = int(summary[-1])

                        #check if the day is an acceptible range
                        if day < 1 or day > 31:
                            print(f"ERROR - ELO summary had 2 items that suggest month day, but day was out of bounds. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                            continue
                        #if it is
                        else:
                            try:
                                #convert to datetime month day
                                if len(summary[0]) > 3:
                                    elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %d').replace(year=predictions_and_summaries.loc[idx,'visit_date'].year)
                                else:
                                    elo_date = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%b %d').replace(year=predictions_and_summaries.loc[idx,'visit_date'].year)

                                #if this proposed date is > the visit date, then it must be from last year
                                if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                    elo_date = elo_date.replace(year = predictions_and_summaries.loc[idx,'visit_date'].year - 1)

                                #this final date is the last occurrence
                                predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date

                            except:
                                print(f"ERROR - ELO summary had 2 items that suggest month day, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                continue
                    #else, there must have been a formatting error
                    else:
                        print(f"ERROR - ELO summary had 2 items, but could not find a suitable timeframe pair. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                        continue

                #if the date is given as three items, it must be in for month, day, year
                elif len(summary) == 3:
                    try:
                        #check if it's an "X or Y" format. If so, take the last one
                        if summary[1].lower() == 'or':
                            if summary[-1].isnumeric():
                                #check if the year has 4 digits
                                if len(summary[-1]) != 4:
                                    print(f"ERROR - ELO summary had 1 item that suggests a year, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                    continue

                                #if it looks ok, convert it to a datetime. Python defaults the date to Jan 1st!!!!
                                summary = datetime.strptime(summary[-1], '%Y')

                                #check if the year is within a reasonable range [1850-today]
                                if summary > datetime.today() or summary < datetime(year=1850, month=1, day=1):
                                    print(f"ERROR - ELO summary had 1 item that suggests a year, but it was out of bounds. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                                    continue

                                #the year is then the last occurrence.
                                predictions_and_summaries.loc[idx,  'last_occurrence'] = summary
                            #if it isn't a number, then it is a single month or single day
                            else:
                                #try to convert it as a month
                                if len(summary[-1]) > 3:
                                    elo_date = datetime.strptime(summary[-1], '%B').replace(year = predictions_and_summaries.loc[idx,'visit_date'].year)
                                else:
                                    elo_date = datetime.strptime(summary[-1], '%b').replace(year = predictions_and_summaries.loc[idx,'visit_date'].year)
                                #if this proposed date is > the visit date, then move the year back by 1
                                if elo_date > predictions_and_summaries.loc[idx,'visit_date']:
                                    elo_date = elo_date.replace(year = predictions_and_summaries.loc[idx,'visit_date'].year - 1)
                                predictions_and_summaries.loc[idx,  'last_occurrence'] = elo_date
                        elif len(summary[-1]) == 4:
                            predictions_and_summaries.loc[idx,  'last_occurrence'] = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %d %Y')
                        else:
                            predictions_and_summaries.loc[idx,  'last_occurrence'] = datetime.strptime(predictions_and_summaries.loc[idx, 'summarization'], '%B %d %y')
                    except:
                        print(f"ERROR - ELO summary had full 3-part date, but could not translate. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                        continue
                else:
                    print(f"ERROR - ELO summary had more than 3 parts. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
                    continue

                #if the ELO could not be determined, warn the user
                if predictions_and_summaries.loc[idx,  'last_occurrence'] == -1:
                    print(f"ERROR - ELO could not be translated. Summary: {predictions_and_summaries.loc[idx, 'summarization']}. Filename: {predictions_and_summaries.loc[idx, 'filename']}")

            else:
                print(f"ERROR - unidentifiable id. id: predictions_and_summaries.loc[idx, 'id']. Filename: {predictions_and_summaries.loc[idx, 'filename']}")
        except Exception as e:
            print(f"ERROR - some problem occurred, skipping. Filename: {predictions_and_summaries.loc[idx, 'filename']}. Summary: {predictions_and_summaries.loc[idx, 'summarization']}\nException: {e}")
            continue
    
#go through and process since last visits
#sort predictions_and_summaries by pat_id and then by visit_date
def process_since_last_visit(predictions_and_summaries):
    """
    Translates "since last visit" frequencies into floats of units per month by attempting to identify a patient's last visit
    Input:
        predictions_and_summaries: pd.DataFrame that was just passed into translate_summaries()
    Returns:
        pd.DataFrame
    """
    
    predictions_and_summaries = predictions_and_summaries.sort_values(['pat_id', 'visit_date'])
    for idx in range(len(predictions_and_summaries)):
        try:
            #skip non since-last-visit pqfs
            if predictions_and_summaries.iloc[idx]['sz_per_month'] != 'calc_since_last_visit_pqf()':
                continue

            #if the summary is a since last visit, get the date of the last visit. Because rows are sorted by visit date, it should be the row directly above
            #first, check that the (above) index is within bounds
            if idx - 1 < 0:
                print(f"Error: previous index out of bounds. idx: {idx}")
                predictions_and_summaries.loc[predictions_and_summaries.iloc[idx].name, 'sz_per_month'] = -2
                continue
            #check if the row above is of the right patient
            if not (predictions_and_summaries.iloc[idx-1]['pat_id'] == predictions_and_summaries.iloc[idx]['pat_id']):
                print(f"Error: previous index is a different patient. idx: {idx}")
                predictions_and_summaries.loc[predictions_and_summaries.iloc[idx].name, 'sz_per_month'] = -2
                continue
            #get the previous visit date and the time passed since then (in months)
            time_passed = (predictions_and_summaries.iloc[idx]['visit_date'] - predictions_and_summaries.iloc[idx-1]['visit_date']).days / 30.4167

            if time_passed == 0:
                print(f"Error: no time difference between current and last visit. idx: {idx}")
                continue

            #calculate the frequency dx/dt
            summary = predictions_and_summaries.iloc[idx]['summarization'].split()
            #if it isn't a number, it's probably a range. Split it by a dash
            if not summary[0].isnumeric(): 
                summary[0] = summary[0].split('-')[-1]

                #if it still isn't a number, it could be an "or", in which case split it by '/'
                if not summary[0].isnumeric(): 
                    summary[0] = summary[0].split('/')[-1]

            predictions_and_summaries.loc[predictions_and_summaries.iloc[idx].name, 'sz_per_month'] = float(summary[0]) / time_passed
        except:
            print(f"ERROR COULD NOT PROCESS THIS SINCE_LAST_VISIT: {predictions_and_summaries.iloc[idx]['summarization']}")
            continue
    return predictions_and_summaries

def process_calendar(unprocessed_text):
    """
    Detects if a string is a seizure calendar by checking if:
        it involves at least two months
        there are at least 2 numbers of seizures
        the associated number of seizures is always 1 word-index afterwards, or within 3 word spaces postwards.
    Input:
        unprocessed_text: a string - generally the output of the finetuned seizure frequency extraction model
    Returns:
        None if it is not a seizure calendar,
        or
        The seizure frequency as a float, with units "per month", if it is a seizure calendar.
    """
    #the abbreviations for months
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    #remove punctuation and split the text by spaces 
    #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    text = unprocessed_text.replace('.', ' . ').replace(',', ' , ').replace('-', ' - ').translate(str.maketrans('','', string.punctuation)).split()
    
    #what are the months in the text, from 1-12?
    month_idx = []

    #this dictionary maps a month idx/position in the text to its associated number idx/position
    month_to_number = {}
    #this dictionary maps a number idx/position in the text to its associated month idx/position
    number_to_month = {}

    #for each word in text, check if it has a number, or a month
    for j in range(len(text)):
        if any(char.isdigit() for char in text[j]): #check if this word has a number in it. https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
            number_to_month[j] = -1
            text[j] = "".join([char for char in text[j] if char.isdigit()])
        for i in range(len(months)):
            if months[i] in text[j].lower():
                month_idx.append(i)
                month_to_number[j] = -1
                
    #check #1b: check for at least 2 months, and at least 2 numbers
    if (len(month_to_number) < 2) and (len(number_to_month) < 2):
        return None
    
    #forward pass: for each month, assume its number of seizures directly follows in the text (i.e. month at idx 6, number at idx 7)
    #for each month
    for month in month_to_number:
        #check if a number occurs at the next index and if that number has not already been taken by another month
        if (month+1) in number_to_month:
            if number_to_month[month+1] == -1:
                
                #check if this number is a year (4 digits > 1970)
                if float(text[month+1]) >= 1970:
                    number_to_month[month+1] = -2
                    
                    #if it is a year, then look at month+2.
                    if (month+2) in number_to_month:
                        if number_to_month[month+2] == -1:
                            month_to_number[month] = month+2
                            number_to_month[month+2] = month
                            
                #if not, assign the month to the number and vice versa
                else:
                    month_to_number[month] = month+1
                    number_to_month[month+1] = month

    #second pass: for each month, if it hasn't been assigned a number, look up to 3 indices backwards for a number.
    #if any of those 3 indices is a number and already taken by a different month, break for this month.
    for month in month_to_number:
        #ignore months that have been assigned a number
        if month_to_number[month] != -1:
            continue
        #look 3 indices backwards for numbers
        for i in range(month-1, month-4, -1):
            #prevent going below index 0
            if i < 0:
                break
            #if it is a number
            if i in number_to_month:
                #check if this number is taken. 
                #If so, break, because you likely would not have a sentence like "6, 7 in May" 
                #if the 7 was already taken by a different month, but 6 was left open
                #if the number is open, then assign it to the month
                if number_to_month != -1:
                    break
                else:
                    number_to_month[i] = month
                    month_to_number = i
                    break

    #check #2: check that at least two months have an associated number
    months_with_associations = []
    total_num_seizures = 0
    for month in month_to_number:
        if month_to_number[month] != -1: #if this month has an associated number
            #note the month, and accumulate the number of seizures
            months_with_associations.append(month)
            total_num_seizures += float(text[month_to_number[month]])
    if len(months_with_associations) < 2:
        return None
    
    print(f"Seizure Calendar detected: '{unprocessed_text}'. Summarization: {total_num_seizures} per {len(months_with_associations)} months")
    return total_num_seizures / len(months_with_associations)


class Patient:
    """Patient object. Contains information about the patient's identifiers, visits and medications"""
    def __init__(self, pat_id):
        self.pat_id = pat_id
        self.visits = []
        self.medications = []
        self.epiType = None
        
    def add_visit(self, visit):
        if visit not in self.visits:
            self.visits.append(visit)
        else:
            print("Warning: This visit already exists for this patient. Visit was not added to Patient's visit list")
            
    def add_medication(self, medication):
        self.medications.append(medication)
        
    def __eq__(self, other):
        if isinstance(other, Patient):
            return self.pat_id == other.pat_id
        else:
            return False
        
    def __str__(self):
        return self.pat_id
    
    
class Aggregate_Patient(Patient):
    """Aggregate Patient Object. Contains information about a patient's identifiers, medications, and aggregate visits."""
    def __init__(self, pat_id):
        super().__init__(pat_id)
        self.aggregate_visits = []
        self.visits = None #aggregate patients must use aggregate_visits.
        
    def add_aggregate_visit(self, aggregate_visit):
        if aggregate_visit not in self.aggregate_visits:
            self.aggregate_visits.append(aggregate_visit)
        else:
            print("Warning: This aggregate visit already exists for this patient. Aggregate Visit was not added to Patient's aggregate visit list")
        
    def __eq__(self, other):
        if isinstance(other, Aggregate_Patient):
            return self.pat_id == other.pat_id
        else:
            return False
        
class Visit:
    """
    Visit Object. Generated from information from a single medical note
        Patient: The Patient with this Visit
        note_id: The note's ID
        pat_enc_id: The visit's patient encounter ID
        author: The name of the provider
        visit_date: The date of the visit
        hasSz: The seizure freedom classification
        pqf: The seizure frequency value
        elo: The date of last seizure value
        context: The note text of the visit
        full_text: The full text of the visit
        visit_type: The type of the visit (new patient, return patient)
    """
    def __init__(self, patient, note_id, pat_enc_id, 
                 author, visit_date, visit_type,
                 hasSz, pqf, elo, 
                 context, full_text):
        
        self.Patient = patient
        self.note_id = note_id
        self.pat_enc_id = pat_enc_id
        self.author = author
        self.visit_date = visit_date
        self.visit_type = visit_type
        
        self.hasSz = hasSz
        self.pqf = pqf
        self.elo = elo
        
        self.context = context
        self.full_text = full_text            
        
    def __str__(self):
        """Prints information for this visit"""
        return f"Visit for patient {self.Patient.pat_id} on {self.visit_date}, written by {self.author}: HasSz = {self.hasSz}; pqf_per_month = {self.pqf}; elo = {self.elo}"
    
    def __eq__(self, other):
        if isinstance(other, Visit):
            return (self.Patient == other.Patient) and (self.note_id == other.note_id) and (self.visit_date == other.visit_date) and (self.author == other.author) and (self.pat_enc_id == other.pat_enc_id)
        else:
            return False
        
class Aggregate_Visit:
    """
    Class for a visit that combines multiple of the same visit (if multiple models with different seeds make predictions for the same visit)
        Aggregate_Patient: The Aggregate Patient with this Visit
        note_id: The visit's ID
        pat_enc_id: The visit's patient encounter ID
        author: The name of the provider
        visit_date: The date of the visit
        all_visits: A list of the (same) visits that make up this single aggregate visit. 
        hasSz: The seizure freedom classification
        pqf: The seizure frequency value
        elo: The date of last seizure value
        context: The note text of the visit
        full_text: The full text of the visit
        visit_type: The type of the visit (new patient, return patient)
    """
    def __init__(self, aggregate_patient, all_visits):
        #first, check if the visits are all the same
        if all_visits.count(all_visits[0]) != len(all_visits):
            raise ValueError(f"Not all visits are the same")
            
        #get the basic info for the visit
        self.Aggregate_Patient = aggregate_patient
        self.all_visits = all_visits
        
        #get information from the visits
        self.note_id = all_visits[0].note_id
        self.pat_enc_id = all_visits[0].pat_enc_id
        self.author = all_visits[0].author
        self.visit_type = all_visits[0].visit_type
        self.visit_date = all_visits[0].visit_date
        
        self.context = all_visits[0].context
        self.full_text = all_visits[0].full_text
    
        #get the hasSz, pqf and elo for each visit
        self.all_hasSz = [vis.hasSz for vis in all_visits]
        self.all_pqf = [vis.pqf if not (pd.isnull(vis.pqf) or isinstance(vis.pqf, str)) else -299.0645 for vis in all_visits] #convert nan or strings to an placemarker arbitrary value for aggregate functions (below)
        self.all_elo = [vis.elo if not (pd.isnull(vis.elo) or isinstance(vis.pqf, str)) else -299.0645 for vis in all_visits] #convert nan or strings to an arbitrary placemarker value for aggregate functions (below)
        
        #calculate plurality voting
        self.hasSz = self.__get_aggregate_hasSz()
        self.pqf = self.__get_aggregate_pqf()
        self.elo = self.__get_aggregate_elo()

        
    def __get_aggregate_hasSz(self):
        """ 
        Gets the seizure freedom value for the aggregate visit by (plurality) voting.
        If there is a tie at the highest number of votes,
            If yes and no have the same number of votes, then default to IDK
            If Yes or No has the same number votes as IDK, then default to either Yes or No
        """        
        #count the votes
        votes = dict.fromkeys(set(self.all_hasSz), 0)
        for vote in self.all_hasSz:
            votes[vote] += 1
            
        #get the value(s) with the highest number of votes
        most_votes = -1
        most_vals = []
        for val in votes:
            if votes[val] > most_votes:
                most_votes = votes[val]
                most_vals = []
            if votes[val] >= most_votes:
                most_vals.append(val)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0]
        #otherwise, if 0,1 both have the highest number of visits, then return idk (2)
        elif (0 in most_vals) and (1 in most_vals):
            return 2
        #otherwise, it must be that either 0 and 1 are tied with idk (2). Return either the 0 or 1
        else:
            most_vals.sort() #sort, since IDK is always 2
            return most_vals[0]
        
    def __get_aggregate_pqf(self):
        """
        Calculate the seizure frequency with plurality voting
        If there is a tie at the highest number of votes, there must be either 2 values with 2 votes, or 5 values with 1 vote.
            Return nan if there are two valid values. Otherwise, there must be at least 1 nan. Return the other value
            If there are 5 potential values, then return nan,
        """
        #count the votes
        votes = dict.fromkeys(set(self.all_pqf), 0)
        for vote in self.all_pqf:
            votes[vote] += 1
            
        #get the value(s) with the highest number of votes
        most_votes = -1
        most_vals = []
        for val in votes:
            if votes[val] > most_votes:
                most_votes = votes[val]
                most_vals = []
            if votes[val] >= most_votes:
                most_vals.append(val)
        most_vals = np.array(most_vals)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0] if most_vals[0] != -299.0645 else np.nan
        #otherwise, if there are two values with the most votes.
        elif len(most_vals) == 2:
            #if one of the values is nan, return the other
            if np.sum(most_vals == -299.0645) == 1:
                return most_vals[~(most_vals == -299.0645)][0]
            #if both values are not nan, return nan
            else:
                return np.nan
        #otherwise, it must be that there are multiple possible answers, each with a vote
        #thus, return nan
        else:
            return np.nan
    
    def __get_aggregate_elo(self):
        """
        Calculate the date of last seizure with plurality voting
        If there is a tie at the highest number of votes, there must be either 2 values with 2 votes, or 5 values with 1 vote.
            Return nan if there are two valid values. Otherwise, there must be at least 1 nan. Return the other value
            If there are 5 potential values, then return nan,
        """
        #count the votes
        votes = dict.fromkeys(set(self.all_elo), 0)
        for vote in self.all_elo:
            votes[vote] += 1
            
        #get the value(s) with the highest number of votes
        most_votes = -1
        most_vals = []
        for val in votes:
            if votes[val] > most_votes:
                most_votes = votes[val]
                most_vals = []
            if votes[val] >= most_votes:
                most_vals.append(val)
        most_vals = np.array(most_vals)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0] if most_vals[0] != -299.0645 else np.nan
        #otherwise, if there are two values with the most votes.
        elif len(most_vals) == 2:
            #if one of the values is nan, return the other
            if np.sum(most_vals == -299.0645) == 1:
                return most_vals[~(most_vals == -299.0645)][0]
            #if both values are not nan, return nan
            else:
                return np.nan
        #otherwise, it must be that there are multiple possible answers, each with a vote
        #thus, return nan
        else:
            return np.nan
    
    def __str__(self):
        return f"Aggregate Visit Object for {self.Aggregate_Patient.pat_id} on {self.visit_date}, written by {self.author}. hasSz: {self.hasSz}, pqf: {self.pqf}, elo: {self.elo}"
    
    def __eq__(self, other):
        if isinstance(other, Aggregate_Visit):
            return (self.Aggregate_Patient == other.Aggregate_Patient) and (self.visit_date == other.visit_date) and (self.author == other.author) and (self.all_visits == other.all_visits)
        else:
            return False
        
def aggregate_patients_and_visits(all_pats):
    """Aggregates patients and visits from dictionary of array of patients all_pats, where each key is a different seed"""

    #initialize the array of Aggregate_Patients
    agg_pats = []
    
    #for simplicity, get the first key
    k = list(all_pats.keys())[0]
    
    #create Aggregate_Patients and fill in their Aggregate_Visits
    for i in range(len(all_pats[k])):
        new_Agg_Pat = Aggregate_Patient(all_pats[k][i].pat_id)
        
        #get aggregate visits
        for j in range(len(all_pats[k][i].visits)):
            new_Agg_visit = Aggregate_Visit(aggregate_patient=new_Agg_Pat,
                                            all_visits = [all_pats[seed][i].visits[j] for seed in all_pats.keys()]
                                           )
            new_Agg_Pat.add_aggregate_visit(new_Agg_visit)
        
        agg_pats.append(new_Agg_Pat)
            
    return agg_pats



def get_asm_name(description, name_dict, exc_names, letter_regex=re.compile(r'[^a-zA-Z]+')):
    """
    Extracts the name of an ASM from the description provided.
    Input:
        description: The prescription description
        name_dict: The dictionary of names and generics we're interested in
        exc_names: The names of drugs we want to exclude
    """
    desc_no_sym = re.sub(letter_regex, ' ', description.lower()).strip()
    desc_split = desc_no_sym.split()

    #iterate through the word-bigrams of the split text
    for i in range(1, len(desc_split)):
        test_str = f"{desc_split[i-1]} {desc_split[i]}"
        if not pd.isnull(name_dict[test_str]):
            return name_dict[test_str]

    #iterate through the word unigrams of the split text
    for text in desc_split:
        if text in exc_names:
            return np.nan
        if not pd.isnull(name_dict[text]):
            return name_dict[text]
    
    return np.nan


def get_all_asm_names_from_description(path_to_asm_names, medications, desc_column_name, asm_subset = None, path_to_exclusion_names = None, return_name_dict=False):
    """
    Extracts the name of an ASM from the description provided in the prescription table
    Input:
        path_to_asm_names: brand and generic names of ASMs, as a csv file
        medications: dataframe with prescription records
        desc_column_name: what is the name of the description column in the dataframe?
        asm_subset: which ASMs are we interested in?
        path_to_exclusion_names: which ASMs do we want to exclude?
    """
    letter_regex = re.compile(r'[^a-zA-Z]+')

    asm_names = pd.read_csv(path_to_asm_names)
    #remove symbols and numbers
    asm_names['Brand'] = asm_names['Brand'].apply(lambda x: re.sub(letter_regex, ' ', x.lower()).strip())
    asm_names['Generic'] = asm_names['Generic'].apply(lambda x: re.sub(letter_regex, ' ', x.lower()).strip())
    asm_names['Abbreviation'] = asm_names['Abbreviation'].apply(lambda x: re.sub(letter_regex, ' ', x.lower()).strip())
    #unify 'extended release' to 'xr'
    asm_names['Brand'] = asm_names['Brand'].apply(lambda x: re.sub(r'extended release', 'xr', x.lower()).strip())
    
    #load in the exclusion criteria
    if path_to_exclusion_names is not None:
        exc_names = pd.read_csv(path_to_exclusion_names, header=None)[0].str.lower().to_numpy()
        asm_names = asm_names.loc[~asm_names.Generic.isin(exc_names)]

    #load in inclusion criteria
    if asm_subset is not None:
        asm_names = asm_names.loc[asm_names.Generic.isin(asm_subset)]

    #dictionary to convert the names of brand names to generic drugs
    brand_to_generic = {row.Brand.lower():row.Generic.lower() for idx, row in asm_names.iterrows()}
    #next, we map abbreviation to generic
    brand_to_generic.update({row.Abbreviation.lower():row.Generic.lower() for idx, row in asm_names.iterrows() if row.Abbreviation.lower() != 'xxx'})
    #for code simplicity, we'll also map generic to generic
    brand_to_generic.update({row.Generic.lower():row.Generic.lower() for idx, row in asm_names.iterrows()})
    brand_to_generic = defaultdict(lambda: np.nan , brand_to_generic)
        
    if return_name_dict:
        return medications[desc_column_name].apply(lambda x: get_asm_name(x, brand_to_generic, exc_names, letter_regex)), brand_to_generic
    else:
        return medications[desc_column_name].apply(lambda x: get_asm_name(x, brand_to_generic, exc_names, letter_regex))
