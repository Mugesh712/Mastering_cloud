# Chest X-Ray Radiology — Fundamental Interpretation Guidelines

## Source: Felson's Principles of Chest Roentgenology, 5th Edition / ACR Standards

### Systematic Approach to Chest X-Ray Reading
- Always use a systematic approach: ABCDE method ensures nothing is missed.
- A (Airway): trachea position (midline vs deviated), carina angle, main bronchi patency.
- B (Bones): ribs, clavicles, scapulae, spine — look for fractures, lytic lesions, degenerative changes.
- C (Cardiac): heart size (CTR < 0.5 normal), heart borders, mediastinal contour, aortic knob.
- D (Diaphragm): position, shape, costophrenic angles, free air under diaphragm (pneumoperitoneum).
- E (Everything else): lung fields (systematically compare left vs right), soft tissues, tubes/lines, extrathoracic structures.
[Source: Felson's Principles of Chest Roentgenology, 5th Edition, Chapter 1]

### Common Radiological Patterns
- Air-space opacification: fluffy, ill-defined borders, air bronchograms present. DDx: pneumonia, pulmonary edema, hemorrhage, ARDS.
- Interstitial pattern: reticular (net-like) or nodular. DDx: interstitial lung disease, viral pneumonia, lymphangitis carcinomatosa.
- Nodular pattern: well-defined round opacities. Single nodule: consider malignancy, granuloma, hamartoma. Multiple nodules: metastases, sarcoidosis, infection.
- Cavitary lesion: thick-walled (> 4mm) suggests abscess or carcinoma. Thin-walled suggests cyst or bulla.
- Linear opacity: atelectasis (volume loss + shift of structures toward opacity), Kerley B lines (pulmonary edema).
[Source: Fundamentals of Diagnostic Radiology, Brant & Helms, 2024 Edition]

### Quality Assessment of Chest X-Ray
- Rotation: medial ends of clavicles should be equidistant from spinous processes.
- Inspiration: at least 6 anterior ribs or 10 posterior ribs visible above the diaphragm.
- Exposure/Penetration: thoracic spine should be barely visible through the cardiac silhouette.
- Projection: PA (posterior-anterior) is standard. AP (anterior-posterior) magnifies the heart — specify on report.
- Artifacts: check for external objects (jewelry, buttons, ECG leads) that may simulate pathology.
[Source: ACR-SPR-STR Practice Parameter for Chest Radiography, 2024]

### Comparison and Clinical Correlation
- ALWAYS compare with prior imaging when available — new findings vs stable findings change management entirely.
- Clinical correlation mandatory: imaging findings alone are insufficient for definitive diagnosis.
- If findings are discordant with clinical presentation, recommend further imaging (CT) or alternative diagnosis.
- Temporal progression: rapidly developing opacity (hours) suggests edema or hemorrhage. Slowly progressive (weeks) suggests infection or neoplasm.
[Source: European Society of Radiology, Best Practices in Radiological Reporting, 2024]

### AI-Assisted Radiology — Interpretation Framework
- DenseNet-121 is a widely validated deep learning architecture for chest X-ray classification with demonstrated AUC 0.82-0.94 across multiple disease classes.
- AI predictions should be interpreted as clinical decision support — NOT as standalone diagnostic tools.
- Confidence scores > 90% indicate high model certainty but still require radiologist confirmation.
- Confidence scores 70-90% indicate moderate certainty — additional imaging or clinical correlation recommended.
- Confidence scores < 70% indicate low certainty — findings should be interpreted with caution, consider alternative diagnoses.
- False positive rate varies by disease class: COVID and pneumonia have higher sensitivity but may over-detect in post-infectious changes.
[Source: Rajpurkar et al., CheXNet: Radiologist-Level Pneumonia Detection, Stanford ML Group]
