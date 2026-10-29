"""
reference_data.py - curated reference tables that the source CSVs do not contain.

The Kaggle-style medication list contains no dosage or prescription information.
Rather than inventing dosage numbers, the system uses:

* a PRESCRIPTION FLAG per treatment (indicative classification, see the report
  limitation section - to be verified by a pharmacist), and
* a GENERIC DOSAGE NOTE per flag that never states a number.

Flags: "OTC" (usually sold without prescription), "Rx/OTC" (depends on product
or strength), "Rx" (prescription required), "Self-care" (non-drug measure) and
"Clinical" (procedure / treatment delivered by a professional).
"""

DOSAGE_NOTES = {
    "OTC": "Follow the label for your age group; do not exceed the stated dose. Ask a pharmacist if unsure.",
    "Rx/OTC": "Strength and dose depend on the product. Check the label or ask a pharmacist or doctor first.",
    "Rx": "Prescription medicine: a doctor sets the dose from age, weight and condition. Do not self-medicate.",
    "Self-care": "General self-care measure; stop and seek advice if symptoms get worse.",
    "Clinical": "Delivered and decided by a healthcare professional.",
}

FLAG_ICONS = {"OTC": "🟢 OTC", "Rx/OTC": "🟡 Rx/OTC", "Rx": "🔴 Rx", "Self-care": "🟢 Self-care", "Clinical": "🏥 Clinical"}

_RX = """ACE inhibitors|Anticholinergics|Anticonvulsants|Antifungal drugs|Antihypertensive medications|
Antimalarial drugs|Antiretroviral drugs|Antispasmodics|Antithyroid medications|Antiviral drugs|
Antiviral medications|Antibiotics|Oral antibiotics|Topical antibiotics|Beta-blockers|Biologics|Blood thinners|
Bronchodilators|Calcium channel blockers|Cholestyramine|Clot-dissolving medications|Corticosteroids|
Inhaled corticosteroids|Cytoprotective agents|DPP-4 inhibitors|Diazoxide|
Disease-modifying antirheumatic drugs (DMARDs)|Diuretics|Entry inhibitors|Epinephrine|Ergotamine derivatives|
Ethambutol|Fluconazole|Fusion inhibitors|GLP-1 receptor agonists|Glucagon injection|Hormonal treatments|
Immunotherapy|Insulin|Integrase inhibitors|Isoniazid|Isotretinoin|Ketoconazole|Leukotriene modifiers|
Levothyroxine|Metformin|Methotrexate|Muscle relaxants|Nitroglycerin|Oral medications|Preventive medications|
Prokinetics|Protease inhibitors|Pyrazinamide|Rifampin|Sulfonylureas|Systemic medications|Thrombolytic drugs|
Triptans|Ursodeoxycholic acid|IV fluids|IV dextrose|Oxygen therapy"""

_RX_OTC = """Analgesics|NSAIDs|Antiemetic drugs|Medications for nausea|Medications for itching|Antihistamines|
Antidiarrheal drugs|Antifungal Cream|Terbinafine|H2 Blockers|Proton Pump Inhibitors (PPIs)|Phenazopyridine|
Mast cell stabilizers|Ointments|Topical treatments|Urinary analgesics|Decongestants"""

_OTC = """Antacids|Antipyretics|Aspirin|Pain relievers|Cough suppressants|Laxatives|Probiotics|Antiseptics|
Clotrimazole|Coal tar|Glucose tablets"""

_SELF_CARE = """Alcohol cessation|Biofeedback|Candy or juice|Cold compresses|Compression stockings|
Elevating the legs|Exercise|High-fiber diet|Home exercises|Hot and cold packs|Joint protection|Neck braces|
Nutritional support|Occupational therapy|Physical therapy|Vestibular rehabilitation|Warm baths|Warm compresses|
Vaccination"""

_CLINICAL = """Blood transfusions|Platelet transfusions|Canalith repositioning|Joint replacement surgery|
Laser treatments|Liver transplant|Phototherapy|Radioactive iodine|Sclerotherapy|Surgery|Thyroid surgery"""


def _split(block: str):
    return [x.strip() for x in block.replace("\n", "").split("|") if x.strip()]


PRESCRIPTION_FLAGS = {}
for _flag, _block in (("Rx", _RX), ("Rx/OTC", _RX_OTC), ("OTC", _OTC), ("Self-care", _SELF_CARE), ("Clinical", _CLINICAL)):
    for _name in _split(_block):
        PRESCRIPTION_FLAGS[_name.lower()] = _flag


def prescription_flag(name: str) -> str:
    """Unknown treatments default to the safest label, 'Rx' (ask a doctor)."""
    return PRESCRIPTION_FLAGS.get(name.strip().lower(), "Rx")


# ---------------------------------------------------------------------------
# Allergy matching: allergen word -> substrings of treatment names to warn about
# ---------------------------------------------------------------------------
ALLERGY_MAP = {
    "penicillin": ["antibiotic"], "amoxicillin": ["antibiotic"], "antibiotic": ["antibiotic"],
    "antibiotics": ["antibiotic"],
    "aspirin": ["aspirin", "nsaid", "pain reliever", "analgesic"],
    "ibuprofen": ["nsaid", "pain reliever", "analgesic"],
    "nsaid": ["nsaid", "aspirin", "pain reliever", "analgesic"],
    "nsaids": ["nsaid", "aspirin", "pain reliever", "analgesic"],
    "paracetamol": ["antipyretic", "pain reliever", "analgesic"],
    "acetaminophen": ["antipyretic", "pain reliever", "analgesic"],
    "sulfa": ["sulfonylurea", "antibiotic"], "sulpha": ["sulfonylurea", "antibiotic"],
    "codeine": ["analgesic", "pain reliever"],
    "antihistamine": ["antihistamine"], "antihistamines": ["antihistamine"],
    "steroid": ["corticosteroid"], "steroids": ["corticosteroid"],
}


def allergy_conflicts(allergens: list[str], treatment_name: str) -> list[str]:
    """Return the allergens that may conflict with this treatment."""
    low = treatment_name.lower()
    hits = []
    for a in allergens:
        a = a.lower().strip()
        patterns = ALLERGY_MAP.get(a, [a])  # unknown allergen: match its own name
        if any(p and p in low for p in patterns):
            hits.append(a)
    return hits
