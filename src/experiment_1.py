import glob
import LR

_EX_NUMBER = "ex1"
_SEED = 42
_OUTPUT_PATH = "data/output/predictions"

# 1. BASELINE on SIMPLE - fastText

EMBEDDINGS_FOLDER = "data/embeddings/fasttext/simple/"
DATASET_FOLDER = "data/data_set/"
X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*train*"))
y_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*simple_train*"))

X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*test*"))
y_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*simple_test*"))

model_name = "fastText"
split_name = "simple"

LR.main(_SEED,
        X_train_files, y_train_files, X_test_files, y_test_files,
        _OUTPUT_PATH, "", model_name, split_name, _EX_NUMBER)


# 2. BASELINE on PSEUDO - fastText

EMBEDDINGS_FOLDER = "data/embeddings/fasttext/pseudo/"
DATASET_FOLDER = "data/data_set/"
X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*train*"))
y_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*simple_train*"))

X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*test*"))
y_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*simple_test*"))

model_name = "fastText"
split_name = "pseudo"

LR.main(_SEED,
        X_train_files, y_train_files, X_test_files, y_test_files,
        _OUTPUT_PATH, "", model_name, split_name, _EX_NUMBER)