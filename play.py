from config import SLIDES_PATH
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.robust_kfold_splitter import RobustPatientWiseKFoldSplitter


# splitter = RobustPatientWiseKFoldSplitter(
#         dataset_dir=SLIDES_PATH,
#         n_splits=5,
#         max_imbalance_ratio=1.3  # Maximum 1.3x class ratio allowed
#     )

# splitter.print_summary()