import glob
import embedding



    
model = "dbmdz/bert-base-italian-cased"
prefix = "BERT"
tokenizer_path = "data/tokenizer"



# # 1. SIMPLE FULL 26/11

# output_path = "data/embeddings/bert/simple/full"
# split = "simple"
# perturbed ="no"
# PATH_INPUT_TEST = "data/data_set/ex_1/simple/full/"
# PATH_INPUT_TRAINING = "data/data_set/ex_1/simple/full/"
# test_files = sorted(glob.glob(PATH_INPUT_TEST+"*simple_test*"))
# train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*simple_train*"))

# embedding.main(model,
#         prefix, tokenizer_path, train_files,test_files,
#         output_path, split , perturbed)

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


# # 2. PSEUDO FULL 26/11

# output_path = "data/embeddings/bert/pseudo/full"
# split = "pseudo"
# perturbed ="no"
# PATH_INPUT_TEST = "data/data_set/ex_1/pseudo/full/"
# PATH_INPUT_TRAINING = "data/data_set/ex_1/pseudo/full/"
# test_files = sorted(glob.glob(PATH_INPUT_TEST+"*pseudo_test*"))
# train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*pseudo_train*"))

# embedding.main(model,
#         prefix, tokenizer_path, train_files,test_files,
#         output_path, split , perturbed)


# # 1. SIMPLE PERTURBED FULL 26/11

# output_path = "data/embeddings/perturbed/full"
# split = "simple"

# perturbation = ["NNP", "NP", "PN", "PNN"] 

# PATH_INPUT_TEST = "data/data_set/ex_1/perturbed/full/"
# PATH_INPUT_TRAINING = "data/data_set/ex_1/simple/full/"

# for p in perturbation:
#         test_files = sorted(glob.glob(PATH_INPUT_TEST+f"*simple_test*_" +p+ ".csv"))
#         train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*simple_train*"))
        
#         print(test_files)
#         print(train_files)


#         embedding.main(model,
#                 prefix, tokenizer_path, train_files,test_files,
#                 output_path, split , p)

# 1. SIMPLE FULL 26/11

# output_path = "data/embeddings/bert/simple/control_simple/"
# split = "simple"
# perturbed ="no"
# PATH_INPUT_TEST = "data/data_set/ex_1/simple/control_simple/"
# PATH_INPUT_TRAINING = "data/data_set/ex_1/simple/control_simple/"
# test_files = sorted(glob.glob(PATH_INPUT_TEST+"*simple_test*"))
# train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*simple_train*"))

# embedding.main(model,
#         prefix, tokenizer_path, train_files,test_files,
#         output_path, split , perturbed)



output_path = "data/embeddings/bert/semantic/"
split = "semantic"
perturbed ="no"
PATH_INPUT_TEST = "data/data_set/ex_2/simple/full/"
PATH_INPUT_TRAINING = "data/data_set/ex_2/simple/full/"
test_files = sorted(glob.glob(PATH_INPUT_TEST+"*simple_test*"))
train_files = sorted(glob.glob(PATH_INPUT_TRAINING+"*simple_train*"))

embedding.main(model,
        prefix, tokenizer_path, train_files,test_files,
        output_path, split , perturbed)
