# execute this command lines from the source github folder.
# First, execute the following command line: export PYTHONPATH=src:${PYTHONPATH}

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_train_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/train_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_val_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/val_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_test_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/test_questions.h5" -min_token_count 1



# to extract a smaller json with a subset of the datasets.
python src/preprocessing/extract_questions_subset.py -data_path "data/CLEVR_v1.0/questions/train_questions.json"
-out_path "data/CLEVR_v1.0/temp/50000_20000_samples/train_questions.json"


# set of starting words when building vocab:
set of starting words...: ['The', 'Does', 'There', 'What', 'How', 'Are', 'Is', 'Do']

