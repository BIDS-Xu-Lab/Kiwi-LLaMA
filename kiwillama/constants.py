def NER_prompt() -> str:
    return """### Task:
Your task is to generate an HTML version of an input text, using HTML <span> tags to mark up specific entities.

### Entity Markup Guides:
Use <span class="problem"> to denote a medical problem.
Use <span class="treatment"> to denote a treatment.
Use <span class="test"> to denote a test.
Use <span class="drug"> to denote a drug.

### Entity Definitions:
Medical Problem: The abnormal condition that happens physically or mentally to a patient.
Treatment: The procedures, interventions, and substances given to a patient for treating a problem.
Drug: Generic or brand name of a single medication or a collective name of a group of medication.
Test: A medical procedure performed (i) to detect or diagnose a problem, (ii) to monitor diseases, disease processes, and susceptibility, or (iii) to determine a course of treatment.

### Input Text: {} <EOS>
### Output Text:"""


def test_prompt() -> str:
    return """### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="labvalue"> to denote a numeric value or a normal description of the result of a lab test.
Use <span class="reference_range"> to denote the range or interval of values that are deemed as normal for a test in a healthy person.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.
Use <span class="temporal"> to denote a calendar date, time, or duration related to a test.

### Input Text: {} <EOS>
### Output Text:"""


def drug_prompt() -> str:
    return """### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="form"> to denote the form of drug.
Use <span class="frequency"> to denote the frequency of taking a drug.
Use <span class="dosage"> to denote the amount of active ingredient from the number of drugs prescribed.
Use <span class="duration"> to denote the time period a patient should take a drug.
Use <span class="strength"> to denote the amount of active ingredient in a given dosage form.
Use <span class="route"> to denote the way by which a drug, fluid, poison, or other substance is taken into the body.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.
Use <span class="temporal"> to denote a calendar date, time, or duration related to a drug.

### Input Text: {} <EOS>
### Output Text:"""


def problem_prompt() -> str:
    return """### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="uncertain"> to denote a measure of doubt.
Use <span class="condition"> to denote a phrase that indicates the problems existing in a certain situation.
Use <span class="subject"> to denote the person entity who is experiencing the disorder.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.
Use <span class="bodyloc"> to denote the location on the body where the observation is present.
Use <span class="severity"> to denote the degree of intensity of a clinical condition.
Use <span class="temporal"> to denote a calendar date, time, or duration related to a problem.
Use <span class="course"> to denote the development or alteration of a problem.

### Input Text: {} <EOS>
### Output Text:"""


def treatment_prompt() -> str:
    return """### Task:
Your task is to mark up modifier entities related to the entity marked with <span> tag in the input text.

### Entity Markup Guide:
Use <span class="temporal"> to denote a calendar date, time, or duration related to a treatment.
Use <span class="negation"> to denote the phrase that indicates the absence of an entity.

### Input Text: {} <EOS>
### Output Text:"""
