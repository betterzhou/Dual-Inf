import openai
import os, csv
import time
import numpy as np
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore")
import ast


openai.api_key = ''


def get_completion_expert_1(prompt, model="gpt-4-turbo-preview"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message["content"]


def get_completion_expert_2(prompt, model="gpt-4-turbo-preview"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message["content"]


def get_completion_whether_reason_wrong(prompt, model="gpt-4-turbo-preview"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message["content"]



patient_info_1 = f""" 
History
HPI: 51-year M construction worker complains of low back pain that started after he lifted heavy boxes 1 week ago. The pain is 8/10, sharp in nature, radiates to the left thigh and sometimes to the left foot. Pain worsens with movement, cough, and sitting for a long time. It is relieved by lying down and partially by ibuprofen. He denies urinary/stool incontinence or weakness/loss of sensation in the lower extremities. No fever, night sweats, or weight loss. He does report difficulty urinating and incomplete emptying of the bladder for 6 months as well as a 1-year history of intermittent low back pain. The pain is exacerbated by sitting for long periods but is relieved by lying down after ambulation.
ROS: Negative except as above.
Allergies: Penicillin, causes rash.
Medications: Ibuprofen.
PMH: None.
PSH: None.
SH: 1 PPD for 18 years, 1-2 beers on weekends, CAGE 0/4.
FH: Noncontributory.
Physical Examination
Patient is in mild distress due to back pain.
VS: WNL
Back: Mild paraspinat muscle tenderness bilaterally, range of motion WNL, no warmth or erythema.
Extremities: 2+ popliteal, dorsalis pedis, and posterior tibial pulses bilaterally. Hips: Nontender and normal range of motion bilaterally
Neuro: Motor: Strength 515 throughout, including left great toe dorsiflexion. DTRs: 2+ symmetric, Babinski bilaterally. Gait: Normal (including toe-to-heel walking), although he walks with back slightly bent forward. Straight leg raising. bilaterally. Sensation: Intact
"""

prediction_rationale_1 = {
    "Disk herniation": [
        "Low back pain",
        "Pain started after lifting heavy boxes",
        "Pain radiates to left thigh and foot",
        "Pain worsens with movement and is relieved by lying still"
    ],
    "lumbar spinal stenosis": [
        "History of intermittent low backpain and leg pain with ambulation",
        "Walks with back slightly bent forward",
        "Pain resolves with sitting"
    ],
    "Metastatic prostate cancer": [
        "Difficulty urinating",
        "Incomplete emptying of the bladder",
        "Low back pain"
    ]
}


patient_info_2 = f""" 
A 27-year-old man presents to the emergency department after a motor vehicle collision. 
The patient was the front seat unrestrained driver in a head on collision. The patient's echocardiogram (ECG) is notable only for sinus tachycardia. 
His temperature is 99.5 °F (37.5 °C), blood pressure is 107/58 mmHg, pulse is 120/min, respirations are 17/min, and oxygen saturation is 98% on room air. 
The patient is given 2 liters of Ringer lactate solution and morphine. 
Initial workup demonstrates that the patient's pulmonary capillary wedge pressure and troponins are elevated. 
The patient is currently complaining of chest pain. Physical exam is notable for an uncomfortable young man with bruising over his chest wall. 
"""

prediction_rationale_2 = {'myocardial infarction': ["the patient's pulmonary capillary wedge pressure and troponins are elevated", 'the patient is currently complaining of chest pain'],
'myocardial contusion': ["the patient's pulmonary capillary wedge pressure and troponins are elevated",
    'the patient is currently complaining of chest pain',
    'Physical exam is notable for an uncomfortable young man with bruising over his chest wall'],
'pulmonary embolism': ["the patient's pulmonary capillary wedge pressure and troponins are elevated",
    'the patient is currently complaining of chest pain'], 'pneumothorax': ['the patient is currently complaining of chest pain',
    'bruising over his chest wall'],
'aortic dissection': ["the patient's pulmonary capillary wedge pressure and troponins are elevated",
    'the patient is currently complaining of chest pain']
}


patient_info_3 = f""" 
A one-day-old male is evaluated in the hospital nursery for bilious vomiting. 
The patient has urinated multiple times since birth but has not yet passed meconium. 
He was born at 38 weeks gestation to a 36-year-old gravida 3 via vaginal delivery. The pregnancy was complicated by gestational diabetes. 
The patient's mother received routine prenatal care and declined first trimester screening. 
She subsequently underwent a quadruple screen at 15 weeks gestation that demonstrated decreased maternal serum alpha-fetoprotein (AFP), increased beta-hCG, decreased unconjugated estriol, and increased inhibin A. 
Her last ultrasound prior to onset of labor demonstrated an amniotic fluid index (AFI) of 28 cm. The patient's two older siblings are both healthy. 
The patient's temperature is 97.8 °F (36.6 °C), blood pressure is 58/37 mmHg, pulse is 166/min, and respirations are 38/min. 
On physical exam, the patient is in moderate distress. 
His abdomen is firm and distended with hypoactive bowel sounds. 
"""

prediction_rationale_3 = {
'Hirschsprung disease': ['the patient has urinated multiple times since birth but has not yet passed meconium', 'His abdomen is firm and distended with hypoactive bowel sounds'],
'meconium ileus': ['bilious vomiting', 'has not yet passed meconium', 'abdomen is firm and distended with hypoactive bowel sounds'],
'intestinal atresia': ['bilious vomiting', 'has not yet passed meconium', 'abdomen is firm and distended with hypoactive bowel sounds'],
'malrotation with midgut volvulus': ['bilious vomiting', 'has not yet passed meconium', 'abdomen is firm and distended with hypoactive bowel sounds'],
'meconium plug syndrome': ['the patient has urinated multiple times since birth but has not yet passed meconium', 'His abdomen is firm and distended with hypoactive bowel sounds'],
'Duodenal atresia': ['bilious vomiting', 'has not yet passed meconium', 'abdomen is firm and distended with hypoactive bowel sounds']
}


patient_info_4 = f""" 
A 30-year-old woman comes to the physician because she has been unable to conceive for 3 years. 
Analysis of her husband's semen has shown normal sperm counts during this time. The patient also reports episodic pelvic and back pain accompanied by painful diarrhea for 5 years. 
She has about one such episode on average per month for 4-6 days. She has taken ibuprofen for the pain, which has provided some relief. 
Menses have occurred at regular 29-day intervals since menarche at the age of 14 years and last for 7 days. 
She is sexually active with her husband and does not use contraception. Vital signs are within normal limits. 
Pelvic and bimanual examinations are normal; rectal examination is unremarkable. A hysterosalpingogram 6 months prior showed normal results. 
"""

prediction_rationale_4 = {
'endometriosis': ['she has been unable to conceive for 3 years', 'episodic pelvic and back pain accompanied by painful diarrhea for 5 years',
    'about one such episode on average per month for 4-6 days'],
'pelvic inflammatory disease (PID)': ['unable to conceive for 3 years', 'episodic pelvic and back pain accompanied by painful diarrhea for 5 years']
}


patient_info_5 = f""" 
A 6-month-old boy presents to his pediatrician for a wellness examination. The mother reports that her child has difficulty rolling from his back to his front and sitting unsupported. 
The patient is able to smile and furrow his brow normally, but she has noticed that he has a weak cry and suck. 
He was born at 38 weeks gestation via a spontaneous vaginal delivery without any complications. 
The mother said that the patient appeared 'normal' until the past few weeks. On physical exam, his extraocular muscle movements are intact, and a symmetric smile is seen. 
He has symmetric flaccid weakness of both his upper and lower extremities. He also has a bell-shaped chest. 
"""

prediction_rationale_5 = {'myopathies': ['symmetric flaccid weakness of both his upper and lower extremities', 'Deep tendon reflexes are diminished'],
'congenital myasthenia gravis': ['difficulty rolling from his back to his front and sitting unsupported', 'weak cry and suck', 'symmetric flaccid weakness of both his upper and lower extremities', 'diminished deep tendon reflexes'],
'anterior horn cell diseases such as spinal muscular atrophy (SMA)': ['symmetric flaccid weakness of both his upper and lower extremities', 'Deep tendon reflexes are diminished']
}


patient_info_6 = f""" 
An 18-month-old girl is brought to the pediatrician by her mother for vaginal bleeding. 
The mother states that she noticed the bleeding today, which appeared as brown discharge with clots in the patient's diaper. 
The mother denies frequent nosebleeds or easy bruising. She also denies any known trauma. 
She does mention that the patient has been limping and complaining of left leg pain since a fall 2 months ago. 
On physical exam, there are multiple 2-3 cm hyperpigmented patches on the patient's trunk. There is bilateral enlargement of the breasts but no pubic hair. 
The vaginal orifice is normal and clear with an intact hymen. 
A plain radiograph of the left lower leg shows patchy areas of lytic bone and sclerosis within the femoral metaphysis.
"""

prediction_rationale_6 = {'precocious puberty': ['bilateral enlargement of the breasts', 'vaginal bleeding'],
'foreign body': ['the vaginal orifice is normal and clear with an intact hymen', 'the mother denies any known trauma'],
'sexual abuse': ['vaginal bleeding', "brown discharge with clots in the patient's diaper"],
'McCune-Albright Syndrome': ["multiple 2-3 cm hyperpigmented patches on the patient's trunk", 'bilateral enlargement of the breasts but no pubic hair', 'patchy areas of lytic bone and sclerosis within the femoral metaphysis']}


patient_info_7 = f""" 
A 50-year-old man presents to his physician with chronic cough, aching joints in his hips and lower back, and malaise over the past 2 months. 
He describes himself as being  generally healthy  before the cough began. Past medical history includes hypertension. The patient takes clopamide and a multivitamin daily. 
His parents are both well and living in a senior living facility. He does not smoke and only drinks alcohol occasionally. 
During a review of symptoms, the patient described a hiking trip in and around the desert near Phoenix, Arizona, 4 months ago. 
At the office, his temperature is 38.6 °C (101.4 °F), heart rate is 102/min, respirations are 20/min, and blood pressure is 120/82 mm Hg. 
A focused chest exam reveals mild fremetus and dullness with percussion on the right side. A chest X-ray shows a right-sided consolidation of the lower right lung, and a chest CT shows an irregular opacity measuring 3.8 cm x 3.0 cm in the sub-plural region of the right middle lobe, a small right-sided pleural effusion, and mild right-sided hilar adenopathy. 
A lung biopsy is performed to rule out cancer and reveals necrotic granulomatous inflammation with multinucleated giant cells and spherules with endospores in the surrounding tissues. 
The laboratory tests show the following results: Hemoglobin 12.9 mg/dL Leukocyte count 9,300/mm3 Platelet count 167,000/mm3 Erythrocyte sedimentation rate 43 mm/hr Periodic acid-Schiff and silver methenamine Positive Acid-fast stain Negative.
"""

prediction_rationale_7 = {
'pneumonia': ['chronic cough', 'right-sided consolidation of the lower right lung', 'irregular opacity measuring 3.8 cm x 3.0 cm in the sub-plural region of the right middle lobe', 'small right-sided pleural effusion'],
'lung cancer': ['a chest CT shows an irregular opacity measuring 3.8 cm x 3.0 cm in the sub-plural region of the right middle lobe', 'a small right-sided pleural effusion', 'mild right-sided hilar adenopathy'],
'fungal infection': ['necrotic granulomatous inflammation with multinucleated giant cells and spherules with endospores in the surrounding tissues', 'Periodic acid-Schiff and silver methenamine Positive']
}


patient_info_8 = f""" 
A 4-year-old boy is brought to the physician because of a generalized rash for 3 days. During this period, he has had severe itching and was not able to sleep well. 
He has also had fever, headache, and some muscle pain. Five days ago, he had a runny nose which subsided with over-the-counter medications. He returned from a camping trip 1 week ago. 
He attends a daycare center. The patient is at the 55th percentile for height and at the 50th percentile for weight. His temperature is 38.2 °C (100.7 °F), pulse is 97/min, and blood pressure is 96/60 mm Hg. 
Examination of the skin shows several macules, papules, and crusted lesions over his face, trunk, and extremities. There are a few fluid-filled vesicles over his abdomen and back. There is no cervical lymphadenopathy. 
The remainder of the examination shows no abnormalities. 
"""

prediction_rationale_8 = {
'measles (rubeola)': ['generalized rash for 3 days', 'severe itching', 'fever', 'headache', 'muscle pain', 'runny nose'],
'varicella (chickenpox)': ['generalized rash for 3 days', 'severe itching', 'fever, headache, and some muscle pain', 'macules, papules, and crusted lesions over his face, trunk, and extremities', 'fluid-filled vesicles over his abdomen and back'],
'allergic reaction': ['generalized rash for 3 days', 'severe itching and was not able to sleep well']
}


patient_info_9 = f""" 
A 65-year-old man presents with a 6-month history of repeated falls, postural dizziness, progressive fatigue, generalized weakness and a 13.6 kg (30 lb) weight loss. He is a vegetarian. 
The patient's family says that he had been high functioning but has had significant behavioral changes over the past year. The patient denies any smoking history, alcohol consumption, or illicit drug use. 
No significant family history. His vital signs include: blood pressure 90/50 mm Hg without postural changes, pulse 92/min, respiratory rate 16/min, temperature 37.0 °C (98.6 °F). 
Physical examination reveals a poorly groomed, disheveled, thin man. He is irritable, paranoid, and delusional but denies any hallucinations. An unstable, wide-based ataxic gait is noted. 
Laboratory results are significant for the following: Hb 6.1 g/dL MCV 109 fL Platelets 90,0000/mm3 Total count 3,000/mm3 Reticulocyte count 0.8% A peripheral blood smear demonstrates hypersegmented neutrophils. Anti-intrinsic factor antibodies are negative. 
"""

prediction_rationale_9 = {
'vitamin B12 deficiency': ['a 6-month history of repeated falls, postural dizziness, progressive fatigue, generalized weakness and a 13.6 kg (30 lb) weight loss', 'significant behavioral changes over the past year', 'an unstable, wide-based ataxic gait', 'Hb 6.1 g/dL MCV 109 fL', 'hypersegmented neutrophils'],
'folate deficiency': ['progressive fatigue, generalized weakness and a 13.6 kg (30 lb) weight loss', 'Hb 6.1 g/dL MCV 109 fL', 'A peripheral blood smear demonstrates hypersegmented neutrophils'],
'copper deficiency': ['repeated falls', 'postural dizziness', 'progressive fatigue', 'generalized weakness', 'weight loss', 'significant behavioral changes', 'irritable, paranoid, and delusional', 'unstable, wide-based ataxic gait'],
'neurodegenerative disorders': ['repeated falls', 'postural dizziness', 'progressive fatigue', 'generalized weakness', 'significant behavioral changes', 'irritable, paranoid, and delusional', 'unstable, wide-based ataxic gait'],
'psychiatric conditions': ['He is irritable, paranoid, and delusional but denies any hallucinations']
}


patient_info_10 = f""" 
A 68-year-old man presents with a 6-month history of worsening fatigue and weight loss. He is especially concerned about a recent episode in which he noticed a blurring of his vision and headache; 
he also notes that his  memory is not as good as it used to be,  which his wife confirms. Physical exam is notable only for pallor and general weakness. 
Laboratory studies reveal anemia as well as an elevated erythrocyte sedimentation rate. Serum protein electrophoresis (SPEP) reveals a sharp, narrow spike of monoclonal IgM; serum IgM concentration is 6.3 g/dL. 
"""

prediction_rationale_10 = {
'multiple myeloma': ['worsening fatigue and weight loss', 'anemia as well as an elevated erythrocyte sedimentation rate', 'sharp, narrow spike of monoclonal IgM; serum IgM concentration is 6.3 g/dL'],
"Waldenstrom's macroglobulinemia": ['worsening fatigue and weight loss', 'blurring of his vision and headache', 'memory is not as good as it used to be', 'anemia', 'elevated erythrocyte sedimentation rate', 'sharp, narrow spike of monoclonal IgM', 'serum IgM concentration is 6.3 g/dL'],
'non-Hodgkin lymphoma': ['a 6-month history of worsening fatigue and weight loss', 'a recent episode in which he noticed a blurring of his vision and headache', 'an elevated erythrocyte sedimentation rate', 'Serum protein electrophoresis (SPEP) reveals a sharp, narrow spike of monoclonal IgM; serum IgM concentration is 6.3 g/dL']
}


def expert_1_prediction_v2_support_feedback(disease_num_j, patient_info_j, whether_low_conf_diagnosis, low_confidence_diagnosis_list):
    Question = f"""Based on the patient’s symptom description, please USE STEP-BY-STEP DEDUCTION TO INFER the most likely diagnoses. 
        In other words, list the differential diagnosis. Please state the names of the most likely diseases. 
        Additionally, for each possible disease, explain which pieces of patient information support this diagnosis. 
        Please note that you should use STEP-BY-STEP DEDUCTION TO INFER the diagnoses and interpretations.
        
        Please note that your response format should be a Python dictionary structure. 
        This means the keys of the dictionary are the names of the predicted diseases, 
        and the value corresponding to each key is the list of reasons supporting this prediction.
            "disease 1": ["reason 1", "reason 2"],
            "disease 2": ["reason 1", "reason 2"],
            ...
            "disease N": ["reason 1", "reason 2"]
        Please note that your response should only be a Python dictionary.
        Do not include special characters like ``` or python.
        """

    Prompt_expert1 = f"""
    You are an experienced doctor. 
    Based on the provided patient information```{patient_info_j}```and the question ```{Question}```,
    please infer the most likely diagnoses.
    
    Below, I provide ten complete examples for you to understand the task and reference the response format. 
        Patient information```{patient_info_1}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_1}```
        Patient information```{patient_info_2}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_2}```
        Patient information```{patient_info_3}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_3}```
        Patient information```{patient_info_4}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_4}```
        Patient information```{patient_info_5}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_5}```
        Patient information```{patient_info_6}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_6}```
        Patient information```{patient_info_7}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_7}```
        Patient information```{patient_info_8}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_8}```
        Patient information```{patient_info_9}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_9}```
        Patient information```{patient_info_10}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_10}```
    Please write your response in the corresponding format.
    Please note that the output should only be a dictionary-type data. 
    Do not include any extraneous characters, such as 'python'; only output the content within the {{}}.
    Please note that abbreviations should not appear in all the texts.
    """

    Prompt_expert1_add_feedback = f"""
    You are an experienced doctor. 
    Based on the provided patient information```{patient_info_j}```and the question ```{Question}```,\
    please infer the most likely diagnoses.
    Please note that your previously predicted diagnoses for this patient SEEMS WRONG.\
    The differential diagnosis of the patient DOES NOT SEEM TO CONTAIN ```{low_confidence_diagnosis_list}```. \
    NOTABLY, THIS FEEDBACK MIGHT NOT NECESSARILY BE CORRECT. 
    Please think twice and combine the following instructions to make another prediction.

    Below, I provide ten complete examples for you to understand the task and reference the response format. 
        Patient information```{patient_info_1}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_1}```
        Patient information```{patient_info_2}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_2}```
        Patient information```{patient_info_3}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_3}```
        Patient information```{patient_info_4}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_4}```
        Patient information```{patient_info_5}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_5}```
        Patient information```{patient_info_6}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_6}```
        Patient information```{patient_info_7}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_7}```
        Patient information```{patient_info_8}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_8}```
        Patient information```{patient_info_9}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_9}```
        Patient information```{patient_info_10}``` and the question```{Question}``` as well as the doctor’s diagnoses and interpretation```{prediction_rationale_10}```
    Please write your response in the corresponding format.
    Please note that the output should only be a dictionary-type data. 
    Do not include any extraneous characters, such as 'python'; only output the content within the {{}}.
    Please note that abbreviations should not appear in all the texts.
    """

    if whether_low_conf_diagnosis:
        expert_1_pred = get_completion_expert_1(Prompt_expert1_add_feedback)
    else:
        expert_1_pred = get_completion_expert_1(Prompt_expert1)
    expert_1_pred_dict = eval(expert_1_pred)  # convert str to dict
    return expert_1_pred_dict


def expert_2_infer_syptom_list_from_book_v2(diagnosis_info_j):
    eg_expert2_disease_1 = "appendicitis"
    eg_expert2_disease_1_symptom_list = ["Pain that starts near the navel and moves to the lower right side of the abdomen",
                         "Pain that worsens when coughing, walking, or making other jarring movements",
                         "Nausea and vomiting",
                         "Loss of appetite",
                         "Low-grade fever that may rise as the illness worsens",
                         "Constipation or diarrhea",
                         "A flushed face",
                         "Elevated White Blood Cell Count (Leukocytosis)"]
    eg_expert2_disease_3 = "Alzheimer's disease"
    eg_expert2_disease_3_symptom_list = ["Memory loss",
                                "Challenges in planning or solving problems",
                                "Difficulty completing familiar tasks",
                                "Confusion with time or place",
                                "problems with words in speaking or writing",
                                "Decreased or poor judgment"]

    Prompt_expert2 = f"""
    You are an experienced doctor. 
    Based on the provided disease name```{diagnosis_info_j}```, please USE STEP-BY-STEP DEDUCTION TO RECALL THE MEDICAL KNOWLEDGE YOU LEARNED AND LIST all the representative symptoms, \
    physical examination results, and medical laboratory results for this disease. 
    Below, I provide some examples for you to understand the task and reference the response format. 
    For the disease```{eg_expert2_disease_1}```, all its common clinical manifestations and examination results are as follows```{eg_expert2_disease_1_symptom_list}```
    For the disease```{eg_expert2_disease_3}```, all its common clinical manifestations and examination results are as follows```{eg_expert2_disease_3_symptom_list}```
    
    Please note that the listed elements, e.g., symptoms or medical laboratory results, can support the diagnosis of the disease.  
    
    Please note that your output should have a specific format. 
    Specifically, the output is a Python list object; each symptom in the list is an element. Each symptom should not contain double quotes internally but should be enclosed in double quotes at the beginning and end. 
    Again, please note that the output should only be in a Python list type; the elements of the list are related symptoms, lab test results, etc.

    Do not include any extraneous characters, such as 'python' or ```; only output the content within the {{}}.
    Please note that abbreviations should not appear in all the texts.
    """

    expert_2_book_symptom = get_completion_expert_2(Prompt_expert2)
    if expert_2_book_symptom.startswith("```"):
        expert_2_book_symptom = expert_2_book_symptom[3:]
    if expert_2_book_symptom.endswith("```"):
        expert_2_book_symptom = expert_2_book_symptom[:-3]

    return expert_2_book_symptom


def evaluate_whether_reason_wrong(disease_k, disease_k_gnd_symptom_list, disease_k_reason_j):
    example_disease_symptom_list = ["Malaise",
                                    "Loss of appetite",
                                    "Bloating",
                                    "Nausea"]
    example_disease_symptom_A = "have a jaded appetite"
    example_disease_symptom_B = "knee swelling and pain"

    Prompt_whether_reason_wrong = f"""
    You are an experienced doctor. 
    I need you to determine whether a symptom```{disease_k_reason_j}``` belongs to the clinical manifestations or test results of the disease ```{disease_k}```.
    Specifically, please use your medical knowledge and the provided symptom list ```{disease_k_gnd_symptom_list}``` for the disease ```{disease_k}``` to \
    determine if ```{disease_k_reason_j}``` also belongs to the clinical manifestations or test results of this disease. \
    Please DO IT STEP-BY-STEP.

    Below, I provide two complete examples for you to understand the task and reference the response format. 
    Based on medical knowledge and the symptom list of the disease```{example_disease_symptom_list}```, ```{example_disease_symptom_A}```also belongs to the clinical manifestations or test results of this disease, so the output is 'Yes'. 
    Based on medical knowledge and the symptom list of the disease```{example_disease_symptom_list}```, ```{example_disease_symptom_B}``` does not belong to the clinical manifestations or test results of this disease, so the output is 'No'.

    Please note that the output should be only 'Yes' or 'No'. \
    Do not include any extraneous characters, such as ```; just output the two or three letters.

    """
    whether_valid_symptom = get_completion_whether_reason_wrong(Prompt_whether_reason_wrong)
    if  whether_valid_symptom == 'Yes' or whether_valid_symptom == 'yes':
        return "Yes"
    if whether_valid_symptom == 'No' or whether_valid_symptom == 'no':
        return "No"


def check_wrong_rationales(pred_dict, gnd_dict):
    GPT_work_counts = 0
    pred_dict_refine = copy.deepcopy(pred_dict)
    for disease_k in list(pred_dict.keys()):
        pred_reason_list = pred_dict[disease_k]
        pred_reason_list_refined = copy.deepcopy(pred_reason_list)
        gnd_reason_list = gnd_dict[disease_k]   

        for disease_k_reason_j in pred_reason_list:
            whether_valid_symptom = evaluate_whether_reason_wrong(disease_k, gnd_reason_list, disease_k_reason_j)
            GPT_work_counts += 1
            if GPT_work_counts % 10 == 0:
                time.sleep(60)  

            if whether_valid_symptom == "No":
                pred_reason_list_refined.remove(disease_k_reason_j)  
        pred_dict_refine[disease_k] = pred_reason_list_refined
    return pred_dict_refine


def evaluate_whether_reason_missing(disease_k, patient_EHR, disease_k_reason_j):
    eg_prompt_EHR_patient = "An 8-year-old boy is brought to the physician by his foster mother because of complaints \
    from his teachers regarding poor performance at school for the past 8 months. He does not listen to their instructions, \
    often talks during class, and rarely completes his school assignments. \
    He does not sit in his seat in the classroom and often cuts in line at the cafeteria. \
    His foster mother reports that he runs around a lot inside the house and refuses to help his sister with chores and errands. \
    He frequently interrupts his foster mother's conversations with others and talks excessively. \
    She has found him trying to climb on the roof on multiple occasions. He was placed in foster care \
    because of neglect by his biological parents 3 years ago. Physical examination shows no abnormalities. \
    Neurologic examination shows no focal findings. Mental status examination shows a neutral affect. "

    eg_prompt_disease_symptom_A = "talks excessively"
    eg_prompt_disease_symptom_B = "do noy follow parent's words"
    eg_prompt_disease_symptom_C = "Low-grade fever"
    eg_prompt_disease_symptom_D = "A flushed face"

    Prompt_whether_reason_missing = f"""
    You are an experienced doctor. 
    I need you to determine whether the patient’s condition description (electronic health record) ```{patient_EHR}``` mentions the clinical manifestations or test results of the disease ```{disease_k}```.
    
    Specifically, please use your medical knowledge to determine if a symptom ```{disease_k_reason_j}``` of the disease```{disease_k}``` appears in this patient’s condition description ```{patient_EHR}```.\
    In other words, does the patient’s condition description```{patient_EHR}``` mention this symptom```{disease_k_reason_j}```? Please USE STEP-BY-STEP DEDUCTION TO DO IT.
    
    Please note, this is not asking for an exact word-for-word match, but rather whether the patient’s condition description```{patient_EHR}```includes clinical manifestations similar to ```{disease_k_reason_j}```.
    
    Below, I provide several complete examples for you to understand the task and reference the response format. 
    Based on medical knowledge, a common symptom of a disease```{eg_prompt_disease_symptom_A}```appears in this patient’s condition description```{eg_prompt_EHR_patient}```, so the output is 'Yes'. 
    Based on medical knowledge, a common symptom of a disease```{eg_prompt_disease_symptom_B}```appears in this patient’s condition description```{eg_prompt_EHR_patient}```, so the output is 'Yes'. 
    Based on medical knowledge, a common symptom of a disease```{eg_prompt_disease_symptom_C}```does not appears in this patient’s condition description```{eg_prompt_EHR_patient}```, so the output is 'No'.
    Based on medical knowledge, a common symptom of a disease```{eg_prompt_disease_symptom_D}```does not appears in this patient’s condition description```{eg_prompt_EHR_patient}```, so the output is 'No'.

    Please note that the output should be only 'Yes' or 'No'. \
    Do not include any extraneous characters, such as ```; just output the two or three letters.

    """
    whether_valid_symptom = get_completion_whether_reason_wrong(Prompt_whether_reason_missing)
    if whether_valid_symptom == 'Yes' or whether_valid_symptom == 'yes':
        return "Yes"
    if whether_valid_symptom == 'No' or whether_valid_symptom == 'no':
        return "No"


def check_missing_rationales_from_EHR(book_dict, patient_note, current_dict):
    GPT_work_counts = 0
    current_dict_refine = copy.deepcopy(current_dict)
    for disease_k in list(book_dict.keys()):
        pred_reason_list = book_dict[disease_k]
        pred_reason_list_refined = copy.deepcopy(pred_reason_list)

        for disease_k_reason_j in pred_reason_list:
            whether_valid_symptom = evaluate_whether_reason_missing(disease_k, patient_note, disease_k_reason_j)
            GPT_work_counts += 1
            if GPT_work_counts % 10 == 0:
                time.sleep(60)  

            if whether_valid_symptom == "Yes":
                pred_reason_list_refined.append(disease_k_reason_j)
        current_dict_refine[disease_k] = pred_reason_list_refined
    return current_dict_refine


def adjust_string(s: str) -> str:
    if s.startswith('["') and s.endswith('"]'):
        return s
    else:
        last_comma_pos = s.rfind('",')
        if last_comma_pos != -1:
            return s[:last_comma_pos + 1] + ']'
        else:
            return "[""]"


def find_low_confidence_diagnosis(pred_dict, threshold=2):
    low_confidence_diagnosis_list = []
    whether_exist_low_confidence_diagnosis = False
    for key, value in pred_dict.items():
        if len(value) <= threshold:
            low_confidence_diagnosis_list.append(key)
            whether_exist_low_confidence_diagnosis = True  # has low conf diag

    return low_confidence_diagnosis_list, whether_exist_low_confidence_diagnosis


def dual_inference_LLM_pred(patient_info_j, disease_num_j, GPT_count_num, threshold=2):
    final_pred_dict = None                # update later
    low_confidence_diagnosis_list = None  # update later
    whether_low_conf_diagnosis = False    # update later
    iter_num = 0

    for k in range(5):
        iter_num += 1
        expert_1_pred_dict = expert_1_prediction_v2_support_feedback(disease_num_j, patient_info_j,
                                                        whether_low_conf_diagnosis, low_confidence_diagnosis_list)

        pred_disease_name_list = list(expert_1_pred_dict.keys())    
        patient_j_final_disease_list = []
        patient_j_final_symptoms_list = []
        for j in range(len(pred_disease_name_list)):
            disease_name_j = pred_disease_name_list[j]
            expert_2_book_symptoms = expert_2_infer_syptom_list_from_book_v2(disease_name_j)  
            expert_2_book_symptoms = adjust_string(expert_2_book_symptoms)  
            try:
                book_symptoms_list = ast.literal_eval(expert_2_book_symptoms)
                patient_j_final_disease_list.append(disease_name_j)
                patient_j_final_symptoms_list.append(book_symptoms_list)
            except (ValueError, SyntaxError):
                print("cannot convert book_symptoms to list !!!!")

        expert_2_pred_dict = dict(zip(patient_j_final_disease_list, patient_j_final_symptoms_list))
        print('**************  expert 2 prediction over  ************** \n')

        GPT_count_num += 6
        if GPT_count_num % 10 == 0:
            time.sleep(60)  

        expert_1_pred_refined_dict = check_wrong_rationales(expert_1_pred_dict, expert_2_pred_dict)

        patient_note = patient_info_j
        dict_form_book = expert_2_pred_dict
        double_refined_dict = check_missing_rationales_from_EHR(dict_form_book, patient_note, expert_1_pred_refined_dict)
        final_pred_dict = double_refined_dict

        low_confidence_diagnosis_list, whether_low_conf_diagnosis = find_low_confidence_diagnosis(final_pred_dict, threshold)
        if not whether_low_conf_diagnosis:
            break

    return final_pred_dict, low_confidence_diagnosis_list, GPT_count_num, iter_num


def main(file_path, columns_of_interest):
    total_disease_num = 0
    total_point_num = 0
    disease_correct_count = 0
    point_correct_count = 0
    df = pd.read_excel(file_path)

    GPT_count_num = 0
    for index, row in df.iterrows():
        patient_info_j, disease_num_j, rationale_num_j, gnd_rationale_j, specialty_j = [row[col_index] for col_index in columns_of_interest]
        gnd_rationale_j = eval(gnd_rationale_j)  
        print(f"\n Patient ID {index}")

        final_pred_dict, low_confidence_diagnosis_list, GPT_count_num, iteration_num = dual_inference_LLM_pred(patient_info_j, disease_num_j, GPT_count_num)

        gnd_disease_list_j = [dis_k for dis_k in gnd_rationale_j.keys()]
        disease_list_j = [dis_j for dis_j in final_pred_dict.keys()]
        one_patient_data = {'patient_info': patient_info_j,
                            'disease_num': int(disease_num_j),
                            'rationale_num': int(rationale_num_j),
                            'interpretation': gnd_rationale_j,
                            'specialty': specialty_j,
                            'gnd_disease': gnd_disease_list_j,
                            'pred_disease': disease_list_j,
                            'pred_rationale': final_pred_dict,
                            'iteration_num': int(iteration_num)}
        try:
            with open(output_file_path, 'r') as f:
                header = False
        except FileNotFoundError:
            header = True
        df_new = pd.DataFrame([one_patient_data])
        df_new.to_csv(output_file_path, mode='a', header=header, index=False, encoding='utf-8-sig')
        print(f"\n ##################### Patient ID {index} finished  #####################")
        GPT_count_num += 2
        if GPT_count_num % 10 == 0:
            time.sleep(60)  


file_path = "./annotated_dataset.xlsx"
output_file_path = "./output.csv"
columns_of_interest = [1, 2, 3, 4, 5]
main(file_path, columns_of_interest)


