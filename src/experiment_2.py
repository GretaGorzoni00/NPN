import glob
import LR

_EX_NUMBER = "ex2"
_SEED = 42
_OUTPUT_PATH = "data/output/"

# per dopo ok per esperimento 2

key_values = ["UNK", "CLS", "PREP"]

EMBEDDINGS_FOLDER = "data/embeddings/bert/semantic/per_dopo/"
DATASET_FOLDER = "data/data_set/ex_2/distractors/full/"

y_train_files = sorted(glob.glob(DATASET_FOLDER+"*train*"))

y_test_files = sorted(glob.glob(DATASET_FOLDER+"*test*"))


model_name = "BERT"
split_name = "per_dopo"

for k in key_values:
        X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*2*distractors_train*.pkl"))
        X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*2*distractors_test*.pkl"))
        print(X_train_files)
        print(X_test_files)
        print(y_test_files)
        print(y_train_files)


        LR.main(_SEED,
                X_train_files, y_train_files, X_test_files, y_test_files,
                _OUTPUT_PATH, k , model_name, split_name, _EX_NUMBER, "", "", solver="lbfgs", label="meaning")



# # 2. BASELINE on semantic

# EMBEDDINGS_FOLDER = "data/embeddings/itwac/pre_lemma_semantic/"
# DATASET_FOLDER = "data/data_set/ex_2/simple/full/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*train*"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*test*"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*test*"))


# model_name = "itwac"
# split_name = "itwac/pre_lemma_semantic"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, "", model_name, split_name, _EX_NUMBER, "", "",solver = "lbfgs", label = "meaning")


# CONTROL CLASSIFIER

# key_values = ["UNK", "CLS", "PREP"]

# EMBEDDINGS_FOLDER = "data/embeddings/bert/semantic/"
# DATASET_FOLDER = "data/data_set/ex_2/simple/control/"

# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*train*"))

# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*test*"))


# model_name = "BERT"
# split_name = "semantic/control"

# for k in key_values:
#         X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*2*simple_train*.pkl"))
#         X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*2*simple_test*.pkl"))
#         print(X_train_files)
#         print(X_test_files)
#         print(y_test_files)
#         print(y_train_files)


#         LR.main(_SEED,
#                 X_train_files, y_train_files, X_test_files, y_test_files,
#                 _OUTPUT_PATH, k , model_name, split_name, _EX_NUMBER, "", "", solver="lbfgs", label="meaning")



# # semantic normale

# key_values = ["UNK", "CLS", "PREP"]

# EMBEDDINGS_FOLDER = "data/embeddings/bert/semantic/"
# DATASET_FOLDER = "data/data_set/ex_2/simple/full/"

# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*train*"))

# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*test*"))


# model_name = "BERT"
# split_name = "semantic"

# for k in key_values:
#         X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*2*simple_train*.pkl"))
#         X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*2*simple_test*.pkl"))
#         print(X_train_files)
#         print(X_test_files)
#         print(y_test_files)
#         print(y_train_files)


#         LR.main(_SEED,
#                 X_train_files, y_train_files, X_test_files, y_test_files,
#                 _OUTPUT_PATH, k , model_name, split_name, _EX_NUMBER, "", "", solver="lbfgs", label="meaning")
