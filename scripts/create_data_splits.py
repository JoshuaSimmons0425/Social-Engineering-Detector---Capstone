"""
Creates six reproducible datasets for the BERT training/evaluation pipeline:

1. training_50_50.csv
2. validation_50_50.csv
3. test_50_50.csv
4. calibration_80_20.csv
5. validation_80_20.csv
6. test_80_20.csv

The source dataset is assumed to be approximately 50:50 benign/malicious.

IMPORTANT:
- The training set remains 50:50.
- The 50:50 validation/test sets remain balanced.
- The 80:20 calibration/validation/test sets are created by
  undersampling the benign class.
- All six sets are mutually exclusive.
- A fixed random seed makes the split reproducible.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("data/gold/final_enriched_dataset.csv")

OUTPUT_DIR = Path("data/splits")

LABEL_COLUMN = "Label"

BENIGN_LABEL = "Benign"
MALICIOUS_LABEL = "Malicious"

RANDOM_SEED = 42

# Training
TRAIN_SIZE = 11_900
TRAIN_PER_CLASS = TRAIN_SIZE // 2

# 50:50 evaluation sets
BALANCED_VALIDATION_SIZE = 4_000
BALANCED_TEST_SIZE = 4_000

BALANCED_VALIDATION_PER_CLASS = 2_000
BALANCED_TEST_PER_CLASS = 2_000

# 80:20 evaluation sets
CALIBRATION_SIZE = 4_000
VALIDATION_80_SIZE = 4_000
TEST_80_SIZE = 4_000

CALIBRATION_BENIGN = 3_200
CALIBRATION_MALICIOUS = 800

VALIDATION_80_BENIGN = 3_200
VALIDATION_80_MALICIOUS = 800

TEST_80_BENIGN = 3_200
TEST_80_MALICIOUS = 800


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sample_class(df, label, n, random_state):
    """
    Randomly sample n observations belonging to a specific class.
    """
    class_df = df[df[LABEL_COLUMN] == label]

    if len(class_df) < n:
        raise ValueError(
            f"Not enough rows for label {label}. "
            f"Requested {n}, available {len(class_df)}."
        )

    return class_df.sample(
        n=n,
        random_state=random_state
    )


def save_split(df, filename):
    """
    Shuffle and save a dataset split.
    """
    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)

    print(f"Saved {filename}: {len(df):,} rows")


def print_distribution(name, df):
    """
    Print class distribution for a dataset.
    """
    counts = df[LABEL_COLUMN].value_counts().to_dict()

    benign = counts.get(BENIGN_LABEL, 0)
    malicious = counts.get(MALICIOUS_LABEL, 0)

    total = benign + malicious

    benign_pct = benign / total * 100
    malicious_pct = malicious / total * 100

    print(
        f"{name:<25} "
        f"{total:>7,} rows | "
        f"Benign: {benign:>6,} ({benign_pct:>5.1f}%) | "
        f"Malicious: {malicious:>6,} ({malicious_pct:>5.1f}%)"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset could not be found:\n{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Loaded dataset: {len(df):,} rows")
print(f"Columns: {len(df.columns)}")


# ============================================================
# VALIDATION
# ============================================================

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        f"Label column '{LABEL_COLUMN}' was not found in the dataset.\n"
        f"Available columns: {list(df.columns)}"
    )

# Remove rows with missing labels
missing_labels = df[LABEL_COLUMN].isna().sum()

if missing_labels > 0:
    raise ValueError(
        f"Found {missing_labels:,} rows with missing labels. "
        "Resolve these before creating the experimental splits."
    )

# Confirm expected labels
unique_labels = set(df[LABEL_COLUMN].unique())

expected_labels = {
    BENIGN_LABEL,
    MALICIOUS_LABEL
}

if unique_labels != expected_labels:
    raise ValueError(
        f"Unexpected label values found: {unique_labels}. "
        f"Expected exactly: {expected_labels}"
    )


# ============================================================
# ORIGINAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL DATASET DISTRIBUTION")
print("=" * 70)

print_distribution("Original dataset", df)

benign_df = df[df[LABEL_COLUMN] == BENIGN_LABEL].copy()
malicious_df = df[df[LABEL_COLUMN] == MALICIOUS_LABEL].copy()

available_benign = len(benign_df)
available_malicious = len(malicious_df)

print(f"\nAvailable benign rows:    {available_benign:,}")
print(f"Available malicious rows: {available_malicious:,}")


# ============================================================
# CHECK DATASET IS LARGE ENOUGH
# ============================================================

required_benign = (
    TRAIN_PER_CLASS
    + BALANCED_VALIDATION_PER_CLASS
    + BALANCED_TEST_PER_CLASS
    + CALIBRATION_BENIGN
    + VALIDATION_80_BENIGN
    + TEST_80_BENIGN
)

required_malicious = (
    TRAIN_PER_CLASS
    + BALANCED_VALIDATION_PER_CLASS
    + BALANCED_TEST_PER_CLASS
    + CALIBRATION_MALICIOUS
    + VALIDATION_80_MALICIOUS
    + TEST_80_MALICIOUS
)

print("\nRequired for experimental splits:")
print(f"Benign:    {required_benign:,}")
print(f"Malicious: {required_malicious:,}")

if available_benign < required_benign:
    raise ValueError(
        f"Insufficient benign examples. "
        f"Need {required_benign:,}, have {available_benign:,}."
    )

if available_malicious < required_malicious:
    raise ValueError(
        f"Insufficient malicious examples. "
        f"Need {required_malicious:,}, have {available_malicious:,}."
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE MUTUALLY EXCLUSIVE SPLITS
# ============================================================

print("\n" + "=" * 70)
print("CREATING SPLITS")
print("=" * 70)


# ------------------------------------------------------------
# STEP 1: Training set
# ------------------------------------------------------------

training_benign = sample_class(
    benign_df,
    BENIGN_LABEL,
    TRAIN_PER_CLASS,
    RANDOM_SEED
)

training_malicious = sample_class(
    malicious_df,
    MALICIOUS_LABEL,
    TRAIN_PER_CLASS,
    RANDOM_SEED
)

training = pd.concat(
    [training_benign, training_malicious]
)

# Remove selected rows from the available pools
benign_remaining = benign_df.drop(training_benign.index)
malicious_remaining = malicious_df.drop(training_malicious.index)


# ------------------------------------------------------------
# STEP 2: 50:50 validation
# ------------------------------------------------------------

validation_50_benign = sample_class(
    benign_remaining,
    BENIGN_LABEL,
    BALANCED_VALIDATION_PER_CLASS,
    RANDOM_SEED + 1
)

validation_50_malicious = sample_class(
    malicious_remaining,
    MALICIOUS_LABEL,
    BALANCED_VALIDATION_PER_CLASS,
    RANDOM_SEED + 1
)

validation_50 = pd.concat(
    [validation_50_benign, validation_50_malicious]
)

benign_remaining = benign_remaining.drop(
    validation_50_benign.index
)

malicious_remaining = malicious_remaining.drop(
    validation_50_malicious.index
)


# ------------------------------------------------------------
# STEP 3: 50:50 test
# ------------------------------------------------------------

test_50_benign = sample_class(
    benign_remaining,
    BENIGN_LABEL,
    BALANCED_TEST_PER_CLASS,
    RANDOM_SEED + 2
)

test_50_malicious = sample_class(
    malicious_remaining,
    MALICIOUS_LABEL,
    BALANCED_TEST_PER_CLASS,
    RANDOM_SEED + 2
)

test_50 = pd.concat(
    [test_50_benign, test_50_malicious]
)

benign_remaining = benign_remaining.drop(
    test_50_benign.index
)

malicious_remaining = malicious_remaining.drop(
    test_50_malicious.index
)


# ------------------------------------------------------------
# STEP 4: 80:20 calibration
# ------------------------------------------------------------

calibration_benign = sample_class(
    benign_remaining,
    BENIGN_LABEL,
    CALIBRATION_BENIGN,
    RANDOM_SEED + 3
)

calibration_malicious = sample_class(
    malicious_remaining,
    MALICIOUS_LABEL,
    CALIBRATION_MALICIOUS,
    RANDOM_SEED + 3
)

calibration = pd.concat(
    [calibration_benign, calibration_malicious]
)

benign_remaining = benign_remaining.drop(
    calibration_benign.index
)

malicious_remaining = malicious_remaining.drop(
    calibration_malicious.index
)


# ------------------------------------------------------------
# STEP 5: 80:20 validation
# ------------------------------------------------------------

validation_80_benign = sample_class(
    benign_remaining,
    BENIGN_LABEL,
    VALIDATION_80_BENIGN,
    RANDOM_SEED + 4
)

validation_80_malicious = sample_class(
    malicious_remaining,
    MALICIOUS_LABEL,
    VALIDATION_80_MALICIOUS,
    RANDOM_SEED + 4
)

validation_80 = pd.concat(
    [validation_80_benign, validation_80_malicious]
)

benign_remaining = benign_remaining.drop(
    validation_80_benign.index
)

malicious_remaining = malicious_remaining.drop(
    validation_80_malicious.index
)


# ------------------------------------------------------------
# STEP 6: 80:20 test
# ------------------------------------------------------------

test_80_benign = sample_class(
    benign_remaining,
    BENIGN_LABEL,
    TEST_80_BENIGN,
    RANDOM_SEED + 5
)

test_80_malicious = sample_class(
    malicious_remaining,
    MALICIOUS_LABEL,
    TEST_80_MALICIOUS,
    RANDOM_SEED + 5
)

test_80 = pd.concat(
    [test_80_benign, test_80_malicious]
)

benign_remaining = benign_remaining.drop(
    test_80_benign.index
)

malicious_remaining = malicious_remaining.drop(
    test_80_malicious.index
)


# ============================================================
# SAVE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("SAVING DATASETS")
print("=" * 70)

save_split(
    training,
    "training_50_50.csv"
)

save_split(
    validation_50,
    "validation_50_50.csv"
)

save_split(
    test_50,
    "test_50_50.csv"
)

save_split(
    calibration,
    "calibration_80_20.csv"
)

save_split(
    validation_80,
    "validation_80_20.csv"
)

save_split(
    test_80,
    "test_80_20.csv"
)


# ============================================================
# FINAL DISTRIBUTION REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT DISTRIBUTIONS")
print("=" * 70)

print_distribution("Training 50:50", training)
print_distribution("Validation 50:50", validation_50)
print_distribution("Test 50:50", test_50)
print_distribution("Calibration 80:20", calibration)
print_distribution("Validation 80:20", validation_80)
print_distribution("Test 80:20", test_80)


# ============================================================
# UNUSED DATA
# ============================================================

unused = pd.concat(
    [benign_remaining, malicious_remaining]
)

print("\n" + "=" * 70)
print("UNUSED DATA")
print("=" * 70)

print_distribution("Unused data", unused)

print(
    "\nThese rows were intentionally not assigned to an experimental split."
)


# ============================================================
# OVERLAP CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FOR OVERLAPPING ROWS")
print("=" * 70)

datasets = {
    "training": training,
    "validation_50": validation_50,
    "test_50": test_50,
    "calibration": calibration,
    "validation_80": validation_80,
    "test_80": test_80,
}

# The dataframe index corresponds to the original row index.
# Verify that no original row occurs in more than one set.

all_indices = []

for name, dataset in datasets.items():
    indices = set(dataset.index)

    overlap = set(all_indices).intersection(indices)

    if overlap:
        raise RuntimeError(
            f"Data leakage detected! "
            f"{name} overlaps with another dataset."
        )

    all_indices.extend(indices)

print("No row overlap detected between experimental datasets.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLITTING COMPLETE")
print("=" * 70)

print(f"\nOutput directory:")
print(f"  {OUTPUT_DIR.resolve()}")

print("\nGenerated files:")

for filename in [
    "training_50_50.csv",
    "validation_50_50.csv",
    "test_50_50.csv",
    "calibration_80_20.csv",
    "validation_80_20.csv",
    "test_80_20.csv",
]:
    print(f"  - {filename}")

print("\nAll splits are reproducible using:")
print(f"  RANDOM_STATE = {RANDOM_SEED}")

print("\nDone.")