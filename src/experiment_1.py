import glob
import LR

_EX_NUMBER = "ex1"
_SEED = 42


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

# # 4. PROBE on SIMPLE - UNK

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple/"
# DATASET_FOLDER = "data/data_set/"
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


# # 12. CONTROL on SIMPLE - UNK - 0.5

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



# # 20. CONTROL on SIMPLE - UNK - 0.7

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple/"
# DATASET_FOLDER = "data/data_set/control/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*train*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*simple_train*0.7*"))

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*simple_test*0.7*"))


# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, "")


# # 20. CONTROL on SIMPLE - UNK - 0.7 SAMPLED 240

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/control/0.7/"
# DATASET_FOLDER_TRAIN = "data/data_set/control/0.7/sampled/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*240*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*simple_train*240*"))

# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/control/0.7/"

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))


# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental = "240"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)

# # 20. CONTROL on SIMPLE - UNK - 0.7 SAMPLED 120

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/control/0.7/"
# DATASET_FOLDER_TRAIN = "data/data_set/control/0.7/sampled/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*120*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*simple_train*120*"))

# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/control/0.7/"

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))


# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental = "120"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)

# # 20. CONTROL on SIMPLE - UNK - 0.7 SAMPLED 120

# EMBEDDINGS_FOLDER_TRAIN = "data/output/embeddings/simple/sampled/control/0.7/"
# DATASET_FOLDER_TRAIN = "data/data_set/control/0.7/sampled/"
# X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAIN+"*train*60*.pkl"))
# y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAIN+"*simple_train*60*"))

# EMBEDDINGS_FOLDER_TEST = "data/output/embeddings/simple/"
# DATASET_FOLDER_TEST = "data/data_set/control/0.7/"

# X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+"*UNK*test*.pkl"))
# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))


# model_name = "BERT"
# split_name = "simple"
# KEY="UNK"
# decremental = "60"

# LR.main(_SEED,
#         X_train_files, y_train_files, X_test_files, y_test_files,
#         _OUTPUT_PATH, KEY , model_name, split_name, _EX_NUMBER, decremental)


# # 20. STUPID CONTROL on SIMPLE 

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple/"
# DATASET_FOLDER = "data/data_set/stupid_control/"
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

# # 25/11 PROBE on SIMPLE - UNK

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple_dataset/simple/"
# DATASET_FOLDER = "data/data_set/simple/"
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

# # 25/11 PROBE on SIMPLE - CLS

# EMBEDDINGS_FOLDER = "data/output/embeddings/simple_dataset/simple/"
# DATASET_FOLDER = "data/data_set/simple/"
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

# # 25/11 PROBE on SIMPLE FULL

# _OUTPUT_PATH = "data/output/metrics/simple/full"

# key_values = ["UNK", "CLS", "PREP"]


# EMBEDDINGS_FOLDER = "data/embeddings/bert/simple/full/"
# DATASET_FOLDER = "data/data_set/ex_1/simple/full/"

# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*simple_train*"))

# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*simple_test*"))

# model_name = "BERT"
# split_name = "simple"

# for k in key_values:
#         X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*train*.pkl"))
#         X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*test*.pkl"))
#         print(X_train_files)
#         print(X_test_files)
#         print(y_test_files)
#         print(y_train_files)


#         LR.main(_SEED,
#                 X_train_files, y_train_files, X_test_files, y_test_files,
#                 _OUTPUT_PATH, k , model_name, split_name, _EX_NUMBER, "")

# # 25/11 PROBE on OTHER FULL
# _OUTPUT_PATH = "data/output/metrics/other/full"

# key_values = ["UNK", "CLS", "PREP"]


# EMBEDDINGS_FOLDER = "data//embeddings/bert/other/full/"
# DATASET_FOLDER = "data/data_set/ex_1/other/full/"

# y_train_files = sorted(glob.glob(DATASET_FOLDER+"*other_train*"))

# y_test_files = sorted(glob.glob(DATASET_FOLDER+"*other_test*"))

# model_name = "BERT"

# split_name = "other"

# for k in key_values:
#         X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*train*.pkl"))
#         X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER+"*"+ k + "*test*.pkl"))



#         LR.main(_SEED,
#                 X_train_files, y_train_files, X_test_files, y_test_files,
#                 _OUTPUT_PATH, k , model_name, split_name, _EX_NUMBER, "")

# # 25/11 PROBE SIMPLE SAMPLED

# _OUTPUT_PATH = "data/output/metrics/simple/sampled"

# decremental_value = ["240", "120", "60"]
# key_values = ["UNK", "CLS", "PREP"]

# EMBEDDINGS_FOLDER_TRAINING = "data//embeddings/bert/simple/sampled/"
# DATASET_FOLDER_TRAINING = "data/data_set/ex_1/simple/sampled/"
# EMBEDDINGS_FOLDER_TEST = "data//embeddings/bert/simple/full/"
# DATASET_FOLDER_TEST = "data/data_set//ex_1/simple/full/"

# y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*simple_test*"))

# model_name = "BERT"
# split_name = "simple"


# for d in decremental_value:

#         y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAINING+"*simple_train*" +d+ "*" ))
        
#         for k in key_values:
                
#                 X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAINING+ "**train*" +d+ "*" +k+ ".pkl"))
#                 X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+ "*" + k + "*test*.pkl"))
        



#                 LR.main(_SEED,
#                         X_train_files, y_train_files, X_test_files, y_test_files,
#                         _OUTPUT_PATH, k , model_name, split_name, _EX_NUMBER, d)
                

# 25/11 PROBE OTHER SAMPLED

_OUTPUT_PATH = "data/output/metrics/other/sampled"

decremental_value = ["240", "120", "60"]
key_values = ["UNK", "CLS", "PREP"]

EMBEDDINGS_FOLDER_TRAINING = "data//embeddings/bert/other/sampled/"
DATASET_FOLDER_TRAINING = "data/data_set/ex_1/other/sampled/"
EMBEDDINGS_FOLDER_TEST = "data//embeddings/bert/other/full/"
DATASET_FOLDER_TEST = "data/data_set/ex_1/other/full/"

y_test_files = sorted(glob.glob(DATASET_FOLDER_TEST+"*other_test*"))

model_name = "BERT"
split_name = "other"


for d in decremental_value:

        y_train_files = sorted(glob.glob(DATASET_FOLDER_TRAINING+"*other_train*" +d+ "*" ))
        
        for k in key_values:
                
                X_train_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TRAINING+ "**train*" +d+ "*" +k+ ".pkl"))
                X_test_files = sorted(glob.glob(EMBEDDINGS_FOLDER_TEST+ "*" + k + "*test*.pkl"))
        


                LR.main(_SEED,
                        X_train_files, y_train_files, X_test_files, y_test_files,
                        _OUTPUT_PATH, k , model_name, split_name, _EX_NUMBER, d)

