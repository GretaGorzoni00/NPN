import glob
import embedding



    
model = "dbmdz/bert-base-italian-cased"
prefix = "BERT"
tokenizer_path = "data/tokenizer"
<<<<<<< HEAD



# 1. SIMPLE FULL 26/11

output_path = "data/embeddings/bert/simple/full"
split = "simple"
perturbed ="no"
PATH_INPUT_TEST = "data/data_set/ex_1/simple/full/"
PATH_INPUT_TRAINING = "data/data_set/ex_1/simple/full/"
=======
output_path = "data/output/embeddings/simple_dataset"


# 1. SIMPLE 

split = "simple"
perturbed ="no"
PATH_INPUT_TEST = "data/data_set/simple/"
PATH_INPUT_TRAINING = "data/data_set/simple/"
>>>>>>> 8b754219b9323d04af46193994616b1450d1c693
test_files = sorted(glob.glob(PATH_INPUT_TEST+"*simple_test*"))
train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*simple_train*"))

embedding.main(model,
        prefix, tokenizer_path, train_files,test_files,
        output_path, split , perturbed)

<<<<<<< HEAD
# # 2. OTHER FULL 26/11

# output_path = "data/embeddings/bert/other/full"
# split = "other"
# perturbed ="no"
# PATH_INPUT_TEST = "data/data_set/ex_1/other/full/"
# PATH_INPUT_TRAINING = "data/data_set/ex_1/other/full/"
# test_files = sorted(glob.glob(PATH_INPUT_TEST+"*other_test*"))
# train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*other_train*"))

# embedding.main(model,
#         prefix, tokenizer_path, train_files,test_files,
#         output_path, split , perturbed)

=======
>>>>>>> 8b754219b9323d04af46193994616b1450d1c693

# # 1. PERTURBED on SIMPLE NNP

# split = "simple"
# perturbed ="NNP"
# PATH_INPUT_TEST = "data/data_set/perturbed/"
# PATH_INPUT_TRAINING = "data/data_set/"
# test_files = sorted(glob.glob(PATH_INPUT_TEST+"*simple_test*NNP*"))
# train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*simple_train*"))

# embedding.main(model,
#         prefix, tokenizer_path, train_files,test_files,
#         output_path, split , perturbed)

