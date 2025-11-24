import glob
import LR

_EX_NUMBER = "ex1"
_SEED = 42
_OUTPUT_PATH = "data/output/predictions/"

# # 1. BASELINE on SIMPLE - fastText

# EMBEDDINGS_FOLDER = "data/embeddings/fasttext/simple/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*train*"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*simple_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*test*"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*simple_test*"))

# model_name = "fastText"
# split_name = "simple"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, "", model_name, split_name, _EX_NUMBER, "")


# # 2. BASELINE on PSEUDO - fastText

# EMBEDDINGS_FOLDER = "data/embeddings/fasttext/pseudo/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*train*"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*test*"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_test*"))

# model_name = "fastText"
# split_name = "pseudo"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, "", model_name, split_name, _EX_NUMBER, "")

# # 3. BASELINE on OTHER - fastText

# EMBEDDINGS_FOLDER = "data/embeddings/fasttext/other/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*train*"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*other_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*test*"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*other_test*"))

# model_name = "fastText"
# split_name = "other"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, "", model_name, split_name, _EX_NUMBER, "")

# 4. PROBE on SIMPLE - UNK

EMBEDDINGS_FOLDER = "data/output/embeddings/simple/"
DATASET_FOLDER = "data/data_set/"
X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*train*.pkl"))
y_train_files = sorted(glob.glob(DATASET_FOLDER+"*simple_train*"))

X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*test*.pkl"))
y_test_files = sorted(glob.glob(DATASET_FOLDER+"*simple_test*"))

model_name = "BERT"
split_name = "simple"
KEY="UNK"

LR.main(_SEED,
        X_train_files, y_train_files, X_test_files, y_test_files,
        _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")

# # 5. PROBE on SIMPLE - CLS

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*CLS*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*simple_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*CLS*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*simple_test*"))

# model_name = "BERT"
# split_name = "simple"
# KEY="CLS"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")

# # 6. PROBE on SIMPLE - PREP

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*PREP*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*simple_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*PREP*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*simple_test*"))

# model_name = "BERT"
# split_name = "simple"
# KEY="PREP"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")

# # 7. PROBE on OTHER - UNK

# EMBEDDINGS_FOLDER = "data/output/embeddings/other/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*other_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*other_test*"))

# model_name = "BERT"
# split_name = "other"
# KEY="UNK"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")

# # 8. PROBE on OTHER - CLS

# EMBEDDINGS_FOLDER =  "data/output/embeddings/other/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*CLS*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*other_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*BERT_embedding_CLS_ex1_other_test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*CLS*test*.pkl"))

# model_name = "BERT"
# split_name = "other"
# KEY="CLS"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")

# # 9. PROBE on OTHER - PREP

# EMBEDDINGS_FOLDER = "data/output/embeddings/other/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*PREP*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*other_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*PREP*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*other_test*"))

# model_name = "BERT"
# split_name = "other"
# KEY="PREP"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")

# # 9. PROBE on PSEUDO - UNK

# EMBEDDINGS_FOLDER ="data/output/embeddings/pseudo/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_test*"))

# model_name = "BERT"
# split_name = "pseudo"
# KEY="UNK"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")

# # 10. PROBE on PSEUDO - CLS

# EMBEDDINGS_FOLDER = "data/output/embeddings/pseudo/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*CLS*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*CLS*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_test*"))

# model_name = "BERT"
# split_name = "pseudo"
# KEY="CLS"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")


# # 11. PROBE on PSEUDO - PREP

# EMBEDDINGS_FOLDER = "data/output/embeddings/pseudo/"
# DATASET_FOLDER = "data/data_set/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*PREP*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*PREP*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*pseudo_test*"))

# model_name = "BERT"
# split_name = "pseudo"
# KEY="PREP"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")


# # 12. CONTROL on SIMPLE - UNK

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple/"
# DATASET_FOLDER = "data/data_set/control/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*simple_train*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*simple_test*"))


# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")


# # 13. PROBE on SIMPLE SAMPLED 240 - UNK

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/"
# DATASET_FOLDER_TRAIN = "data/data_set/sampled/"
# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/"

# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*240.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*train*240*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))


# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental="240"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)

# # 14. PROBE on SIMPLE SAMPLED 120 - UNK

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/"
# DATASET_FOLDER_TRAIN = "data/data_set/sampled/"
# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/"

# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*120.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*train*120*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))


# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental="120"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)

# # 15. PROBE on SIMPLE SAMPLED 60 - UNK

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/"
# DATASET_FOLDER_TRAIN = "data/data_set/sampled/"
# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/"

# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*60.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*train*60*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))



# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental="60"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)

# # 16. CONTROL on SIMPLE SAMPLED 240 - UNK

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/control/"
# DATASET_FOLDER_TRAIN = "data/data_set/control/sampled/"
# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/control/"

# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*240.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*train*240*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))

# print(X_train_files)
# print(y_train_files)
# print(X_test_files)
# print(y_test_files)



# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental="240"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)

# # 18. CONTROL on SIMPLE SAMPLED 120 - UNK

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/control/"
# DATASET_FOLDER_TRAIN = "data/data_set/control/sampled/"
# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/control/"

# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*120.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*train*120*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))



# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental="120"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)

# # 19. CONTROL on SIMPLE SAMPLED 60 - UNK

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/control/"
# DATASET_FOLDER_TRAIN = "data/data_set/control/sampled/"
# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/control/"

# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*60.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*train*60*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))




# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental="60"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)