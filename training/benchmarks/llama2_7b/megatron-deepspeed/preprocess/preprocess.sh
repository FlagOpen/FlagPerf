INPUT_DIR=$1
OUTPUT_DIR=$2
FILES=`ls $INPUT_DIR/*.jsonl | tr '\n' ' ' | sed 's/ /,/g'`
FILES=${FILES:0:-1}

git clone https://githubfast.com/EleutherAI/gpt-neox.git
cd gpt-neox
python tools/datasets/preprocess_data.py \
        --input $FILES \
        --output-prefix $OUTPUT_DIR \
        --vocab ../tokenizer.json \
        --tokenizer-type HFTokenizer \
        --append-eod \
        --jsonl-keys text \
        --workers 64