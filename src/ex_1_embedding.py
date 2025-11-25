import glob
import embedding



    
model = "dbmdz/bert-base-italian-cased"
prefix = "BERT"
tokenizer_path = "data/tokenizer"
output_path = "data/output/embeddings/simple_dataset"


# 1. SIMPLE 

split = "simple"
perturbed ="no"
PATH_INPUT_TEST = "data/data_set/simple/"
PATH_INPUT_TRAINING = "data/data_set/simple/"
test_files = sorted(glob.glob(PATH_INPUT_TEST+"*simple_test*"))
train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*simple_train*"))

embedding.main(model,
        prefix, tokenizer_path, train_files,test_files,
        output_path, split , perturbed)


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

