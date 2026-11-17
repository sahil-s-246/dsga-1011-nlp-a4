import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

def analyze_split(nl_path, sql_path, prefix=""):
    with open(nl_path, "r", encoding="utf-8") as f:
        nl_lines = [prefix + line.strip() for line in f if line.strip()]
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_lines = [line.strip().rstrip(';') for line in f if line.strip()]

    nl_tokens = [tokenizer.encode(s, add_special_tokens=True) for s in nl_lines]
    sql_tokens = [tokenizer.encode(s, add_special_tokens=True) for s in sql_lines]

    nl_vocab = set(token for seq in nl_tokens for token in seq)
    sql_vocab = set(token for seq in sql_tokens for token in seq)

    return {
        "num_examples": len(nl_lines),
        "mean_nl_len": np.mean([len(t) for t in nl_tokens]),
        "mean_sql_len": np.mean([len(t) for t in sql_tokens]),
        "nl_vocab": len(nl_vocab),
        "sql_vocab": len(sql_vocab),
    }

if __name__ == "__main__":
    train_nl = "data/train.nl"
    train_sql = "data/train.sql"
    dev_nl = "data/dev.nl"
    dev_sql = "data/dev.sql"

    # Table 1: Raw (Before Preprocessing)
    t1_train = analyze_split(train_nl, train_sql)
    t1_dev   = analyze_split(dev_nl, dev_sql)

    # Table 2: Preprocessed (Task prefix "translate English to SQL: ")
    prefix = "translate English to SQL: "
    t2_train = analyze_split(train_nl, train_sql, prefix=prefix)
    t2_dev   = analyze_split(dev_nl, dev_sql, prefix=prefix)

    print("=" * 60)
    print("TABLE 1: DATA STATISTICS BEFORE PREPROCESSING")
    print("=" * 60)
    print(f"{'Statistic':<35} | {'Train':<10} | {'Dev':<10}")
    print("-" * 60)
    print(f"{'Number of examples':<35} | {t1_train['num_examples']:<10} | {t1_dev['num_examples']:<10}")
    print(f"{'Mean sentence length':<35} | {t1_train['mean_nl_len']:<10.2f} | {t1_dev['mean_nl_len']:<10.2f}")
    print(f"{'Mean SQL query length':<35} | {t1_train['mean_sql_len']:<10.2f} | {t1_dev['mean_sql_len']:<10.2f}")
    print(f"{'Vocabulary size (natural language)':<35} | {t1_train['nl_vocab']:<10} | {t1_dev['nl_vocab']:<10}")
    print(f"{'Vocabulary size (SQL)':<35} | {t1_train['sql_vocab']:<10} | {t1_dev['sql_vocab']:<10}")

    print("\n" + "=" * 60)
    print("TABLE 2: DATA STATISTICS AFTER PREPROCESSING")
    print("=" * 60)
    print(f"{'Statistic':<35} | {'Train':<10} | {'Dev':<10}")
    print("-" * 60)
    print(f"{'Number of examples':<35} | {t2_train['num_examples']:<10} | {t2_dev['num_examples']:<10}")
    print(f"{'Mean sentence length':<35} | {t2_train['mean_nl_len']:<10.2f} | {t2_dev['mean_nl_len']:<10.2f}")
    print(f"{'Mean SQL query length':<35} | {t2_train['mean_sql_len']:<10.2f} | {t2_dev['mean_sql_len']:<10.2f}")
    print(f"{'Vocabulary size (natural language)':<35} | {t2_train['nl_vocab']:<10} | {t2_dev['nl_vocab']:<10}")
    print(f"{'Vocabulary size (SQL)':<35} | {t2_train['sql_vocab']:<10} | {t2_dev['sql_vocab']:<10}")
    print("=" * 60)

