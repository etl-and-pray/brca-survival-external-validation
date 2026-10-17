"""Project-wide settings. Change the analysis choices here, not in the notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

SEED = 42

# --- Data sources (cBioPortal DataHub) ---------------------------------------
# Public study archives; nothing here requires registration.
# Tried in order; the first that works is used. The old S3 bucket now returns 403.
DATAHUB_URLS = [
    "https://datahub.assets.cbioportal.org/{study}.tar.gz",
    "https://cbioportal-datahub.s3.amazonaws.com/{study}.tar.gz",
]
# Fallback: individual files from the GitHub DataHub repo (git LFS served via media host).
GITHUB_FILE_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/{study}/{file}"
STUDIES = {
    "tcga": "brca_tcga_pan_can_atlas_2018",  # development cohort
    "metabric": "brca_metabric",             # external validation cohort
}
CLINICAL_PATIENT_FILE = "data_clinical_patient.txt"
CLINICAL_SAMPLE_FILE = "data_clinical_sample.txt"
EXPRESSION_FILES = {
    "tcga": "data_mrna_seq_v2_rsem.txt",             # RNA-seq, RSEM
    "metabric": "data_mrna_illumina_microarray.txt",  # Illumina HT-12 microarray (log2 intensities)
}

# --- Analysis choices --------------------------------------------------------
# "OS"  : overall survival (available in both cohorts, includes non-cancer deaths)
# "RFS" : TCGA DFS_* is matched to METABRIC RFS_* (recurrence/disease-free)
ENDPOINT = "OS"
ENDPOINT_COLUMNS = {
    "OS": {"tcga": ("OS_MONTHS", "OS_STATUS"), "metabric": ("OS_MONTHS", "OS_STATUS")},
    "RFS": {"tcga": ("DFS_MONTHS", "DFS_STATUS"), "metabric": ("RFS_MONTHS", "RFS_STATUS")},
}

# METABRIC follow-up is much longer than TCGA. Administrative censoring at a
# common horizon keeps the two cohorts comparable.
HORIZON_MONTHS = 120
EVAL_TIMES_MONTHS = [36, 60, 84]  # time points for AUC(t), Brier, calibration

# Stage is defined differently in the two cohorts (AJCC pathologic stage in TCGA,
# a tumour-stage field in METABRIC). Set False for a cleaner comparison.
USE_STAGE = True

# --- Gene panel --------------------------------------------------------------
# PAM50 with current HGNC symbols (ORC6L->ORC6, KNTC2->NDC80, CDCA1->NUF2).
# A fixed small panel is deliberate: TCGA-BRCA has only ~100 deaths, so using
# thousands of genes would overfit.
# Old HGNC symbols that may still appear in one of the cohorts -> current symbol.
GENE_ALIASES = {"ORC6L": "ORC6", "KNTC2": "NDC80", "CDCA1": "NUF2"}

PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "NAT1", "NDC80", "NUF2", "ORC6", "PGR",
    "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]
