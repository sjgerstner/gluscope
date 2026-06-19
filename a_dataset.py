"""
Preprocess given dataset for neuroscope,
using given tokenizer:
Make tokenized blocks of fixed length.
"""

from argparse import ArgumentParser
import os

from transformers import AutoTokenizer
import datasets

parser = ArgumentParser(description="""
Preprocess given dataset for neuroscope,
using given tokenizer:
Make tokenized blocks of fixed length.
"""
)
parser.add_argument('--tokenizer', type=str, default="allenai/OLMo-1B-hf",
                    help="""
                    ID of a Huggingface model, this model's tokenizer will be applied to the dataset.
                    Note that all first-generation OLMo models (i.e., before OLMo-2) use the same tokenizer,
                    so you don't need to run this script repeatedly.
                    """)
parser.add_argument('--dataset', type=str, default="emozilla/dolma-v1_7-3B",
                    help="""
                    Ideally, a subset of approx. 3B tokens from the model's training data.
                    Alternative subset of Dolma: agentlans/dolma-1m
                    """,
                    )
parser.add_argument('--datadir', type=str, default='datasets',
                    help="""
                    general data directory.
                    The dataset will be saved to a subdirectory based on dataset and model name.
                    """,
                    )
parser.add_argument('--save_to', type=str, default=None,
                    help="""
                    name of the place within args.datadir where the processed dataset should be saved.
                    Default will be based on dataset and model name.
                    """,)
parser.add_argument('--add_bos_token', type=bool, default=True,
                    help="add bos token to every example")
parser.add_argument('--max_length', type=int, default=1024,
                    help="length of example token blocks")
parser.add_argument('--return_overflowing_tokens', type=bool, default=False,
                    help="""Make additional training examples with overflowing tokens.
                    In this case it is currently not possible to keep the ids and metadata.""")
parser.add_argument('--padding', type=bool, default=False,
                    help="pad examples to args.max_length")
group = parser.add_mutually_exclusive_group()
group.add_argument('--Mtokens', type=int, default=20,
                    help="""
                    tokens to save, in millions
                    """,
                    )
group.add_argument('--save_all', action='store_true')
args = parser.parse_args()

dataset = datasets.load_dataset(args.dataset, split='train')
print(dataset)
assert isinstance(dataset, datasets.Dataset)

if not args.save_all:#sample first to avoid wasted compute later
    print('selecting subset...')
    dataset = dataset.shuffle(seed=2512800)
    dataset = dataset.select(range(4000*args.Mtokens))
    #make sure there are still enough samples, later we will filter some of them out

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.add_bos_token = args.add_bos_token

def _tokenization(example):
    return tokenizer(
        example["text"],
        max_length=args.max_length,
        truncation=True,
        return_overflowing_tokens=args.return_overflowing_tokens,
        padding='max_length' if args.padding else False,
        )
#TODO (low prio):
# find a solution to keep id and metadata columns when returning overflowing tokens
dataset = dataset.map(_tokenization,
                 batched=True,
                 remove_columns=dataset.column_names if args.return_overflowing_tokens else None,
                 )
#input_ids, attention_mask, (overflow_to_sample_mapping)

if not args.save_all:
    # if not args.padding:
    #     dataset = dataset.filter(lambda example: len(example['input_ids'])==1024)
    #     print('after removing short examples:', dataset.num_rows)
    print('finding optimal number of rows...')
    cum_tokens=0
    goal = args.Mtokens * 1000000
    num_rows=0
    while cum_tokens<goal:
        cum_tokens += len(dataset[num_rows]['input_ids'])
        num_rows+=1
    print(f'selecting {num_rows} rows...')
    dataset = dataset.select(range(num_rows))
    print(f'after subsampling: {dataset.num_rows} rows, {cum_tokens} tokens in total')

dataset.set_format(type="torch")
if args.return_overflowing_tokens:
    dataset.set_format(columns=['input_ids', 'attention_mask'])

print(dataset[0])
print(tokenizer.decode(dataset[0]['input_ids']))

DATADIR = os.path.join(
    os.environ["WORK"] if "WORK" in os.environ else ".",
    args.datadir
)
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR, exist_ok=True)
if args.save_to:
    SAVE_TO = args.save_to
else:
    dataset_short = args.dataset.split('/')[-1]
    tokenizer_short = "-".join(args.tokenizer.split('/')[-1].split('-')[:2])
    SAVE_TO = f"{dataset_short}-{tokenizer_short}"
dataset.save_to_disk(
    f"{DATADIR}/{SAVE_TO}"
    )
