uv run ./scripts/train_embert.py (
    "--data_dir", "C:\PhD\BERT_Project\emBERT\data\SzegedNER\data\business",
    "--bert_model", "SZTAKI-HLT/hubert-base-cc",
    "--task_name", "szeged_ner_bioes",
    "--data_format", "csv",
    "--output_dir", "bert_np",
    "--do_train",
    "--max_seq_length", "384",
    "--num_train_epochs=4",
    "--train_batch_size", "10",
    "--learning_rate", "1e-5",
    "--do_eval",
    "--eval_batch_size", "1",
    "--use_viterbi",
    "--seed", "42"
)