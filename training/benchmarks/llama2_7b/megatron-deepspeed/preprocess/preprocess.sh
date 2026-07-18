# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
