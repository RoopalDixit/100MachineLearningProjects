# Notes for Rule-Based Medical Diagnosis Assistant (Educational Tool)

## Overview
- **Purpose**: An educational tool to demonstrate a simple symptom-based disease diagnosis system using Python. **Not for real medical use.**
- **Approach**: Rule-based matching where user-input symptoms are compared to a hardcoded dictionary of diseases and their symptoms.
- **Key Features**:
  - Matches user symptoms to diseases based on overlap percentage (>50%).
  - Displays matching diseases, their match scores, and shared symptoms.
  - Simple, no external dependencies required.

## Code Structure
1. **Disease Database**:
   - A Python dictionary (`diseases`) maps disease names to lists of symptoms.
   - Example: `{"Flu": ["fever", "cough", "fatigue", "body aches"], ...}`.
   - Contains 7 diseases with 5-6 symptoms each, based on general knowledge.

2. **User Input**:
   - Uses `input()` to collect comma-separated symptoms (e.g., "fever, cough").
   - Processes input by stripping whitespace and converting to lowercase for case-insensitive matching.

3. **Matching Logic**:
   - Iterates through each disease's symptom list.
   - Calculates overlap: number of user symptoms present in disease symptoms.
   - Computes match score as `(len(matching_symptoms) / len(disease_symptoms)) * 100`.
   - Includes diseases with scores >50% in results.

4. **Output**:
   - Sorts matches by score (descending).
   - Prints each disease, its match percentage, and matching symptoms.
   - If no matches, advises consulting a doctor.

## Key Code Snippets
- **Input Processing**:
  ```python
  user_input = input("Enter your symptoms separated by commas (e.g., fever, cough, fatigue): ")
  user_symptoms = [symptom.strip().lower() for symptom in user_input.split(",")]
  ```
  - Splits input string, removes spaces, and standardizes case.

- **Matching**:
  ```python
  matching_symptoms = [s for s in user_symptoms if s in [sym.lower() for sym in symptoms]]
  match_score = (len(matching_symptoms) / len(symptoms)) * 100
  ```
  - Finds common symptoms and computes percentage match.

## Learning Points
- **Dictionaries**: Efficient for storing and accessing disease-symptom mappings.
- **List Comprehensions**: Used for clean input processing and symptom matching.
- **Logic Flow**: Demonstrates looping, conditionals, and sorting for decision-making.
- **Limitations**:
  - Static data (hardcoded, small dataset).
  - No symptom weighting or fuzzy matching for typos.
  - Basic threshold-based logic, not probabilistic.

## Example Usage
- Input: `"fever, cough, fatigue"`
- Output:
  ```
  Possible diagnoses based on your symptoms (educational only):
  - Flu: 66.7% match
    Matching symptoms: fever, cough, fatigue
  - COVID-19: 66.7% match
    Matching symptoms: fever, cough, fatigue
  - Pneumonia: 60.0% match
    Matching symptoms: cough, fever, fatigue
  ```

## Potential Improvements
- Add fuzzy matching (e.g., `fuzzywuzzy` library) for typo tolerance.
- Expand disease database or load from a file (e.g., CSV).
- Weight symptoms by severity for better prioritization.
- Upgrade to ML for probabilistic predictions (see Version 2).

## Ethical Note
- Includes a disclaimer emphasizing this is not for real medical diagnosis.
- Reinforces the need to consult professionals for health concerns.