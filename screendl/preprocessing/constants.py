""""""

from types import SimpleNamespace

# default file names for processed data
PROCESSED_DATA_FILES = SimpleNamespace()

PROCESSED_DATA_FILES.omics_exp = "OmicsGeneExpressionTPM.csv"
PROCESSED_DATA_FILES.omics_mut = "OmicsSomaticMutations.csv"
PROCESSED_DATA_FILES.omics_cnv = "OmicsTotalCopyNumber.csv"

PROCESSED_DATA_FILES.meta_samples = "SampleAnnotations.csv"
PROCESSED_DATA_FILES.meta_drugs = "DrugAnnotations.csv"
