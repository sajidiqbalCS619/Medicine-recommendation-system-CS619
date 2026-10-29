# Test results

34 of 34 automated tests passed.

## Design Document test cases

| ID | Use case | Scenario | Automated checks | Status |
|---|---|---|---|---|
| TC-01a | UC-01 | Register with a valid, unique email | 2 | **Pass** |
| TC-01b | UC-01 | Register with an already-used email | 1 | **Pass** |
| TC-02a | UC-02 | Login with correct credentials | 2 | **Pass** |
| TC-02b | UC-02 | Login with repeated incorrect password | 1 | **Pass** |
| TC-03a | UC-03 | Get recommendations from valid symptoms | 2 | **Pass** |
| TC-03b | UC-03 | Low-confidence prediction | 2 | **Pass** |
| TC-04a | UC-04 | View history with prior queries | 1 | **Pass** |
| TC-04b | UC-04 | View history with no prior queries | 1 | **Pass** |
| TC-05a | UC-05 | Submit valid feedback | 1 | **Pass** |
| TC-05b | UC-05 | Close feedback form without submitting | 1 | **Pass** |
| TC-06a | UC-06 | Upload a valid dataset | 1 | **Pass** |
| TC-06b | UC-06 | Upload an invalid dataset | 1 | **Pass** |
| TC-07a | UC-07 | Retrain and deploy an improved model | 1 | **Pass** |
| TC-07b | UC-07 | Retrain yields a lower-accuracy model | 1 | **Pass** |

## Additional tests

| Test | Status |
|---|---|
| test_UC01_validation_rules | Pass |
| test_UC03_no_symptom_is_rejected_politely | Pass |
| test_UC03_exception_model_unavailable | Pass |
| test_UC03_age_gender_allergy_read_from_the_sentence | Pass |
| test_UC03_allergy_map_penicillin_flags_antibiotics | Pass |
| test_UC03_child_and_pregnancy_cautions | Pass |
| test_UC03_prescription_flag_and_dosage_note_never_contain_numbers | Pass |
| test_UC05_rating_must_be_1_to_5 | Pass |
| test_UC06_exception_file_over_50mb | Pass |
| test_UC06_json_dataset_accepted | Pass |
| test_UC07_exception_training_crash_keeps_previous_model | Pass |
| test_NFR04_sensitive_fields_are_encrypted_at_rest | Pass |
| test_NFR04_aes256_key_and_roundtrip | Pass |
| test_NFR08_production_model_accuracy_at_least_85_percent | Pass |
| test_NFR09_consent_required_and_recorded | Pass |
| test_NFR_response_time_under_3_seconds | Pass |
