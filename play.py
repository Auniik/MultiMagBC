from config import SLIDES_PATH
from preprocess.kfold_splitter import PatientWiseKFoldSplitter


splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=True
    )

splitter.visualize()