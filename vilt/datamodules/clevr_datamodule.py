import os
import json
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from vilt.datasets import ClevrDataset
from tqdm import tqdm
from .datamodule_base import BaseDataModule


CLASSES = {
    'number':['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'material':['rubber', 'metal'],
    'color':['cyan', 'blue', 'yellow', 'purple', 'red', 'green', 'gray', 'brown'],
    'shape':['sphere', 'cube', 'cylinder'],
    'size':['large', 'small'],
    'exist':['yes', 'no'],
}


def tokenize(sentence):
    # punctuation should be separated from the words
    # import pdb; pdb.set_trace()
    s = re.sub('([.,;:!?()])', r' \1 ', str(sentence))
    s = re.sub('\s{2,}', ' ', s)
    '''
    try:
        s = re.sub('([.,;:!?()])', r' \1 ', sentence)
        s = re.sub('\s{2,}', ' ', s)
    except:
        print(sentence)
        import pdb; pdb.set_trace()
    '''

    # tokenize
    split = s.split()

    # normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower


class ClevrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def dataset_cls(self):
        return ClevrDataset
    
    @property
    def dataset_name(self):
        return "clevr"
    
    def prepare_data(self):
        # prepare tables
        splits = ["train", "val", "test"]
        qst_dir = os.path.join(self.data_dir, "questions")
        tables = []
        for split in splits:
            table_path = os.path.join(self.data_dir, f"{self.dataset_name}_{split}.parquet")
            if not os.path.exists(table_path):
                qst_path = os.path.join(qst_dir, f"CLEVR_{split}_questions.json")
                with open(qst_path, "r") as f:
                    qsts = json.load(f)["questions"]
                    table = pa.Table.from_pylist(qsts)
                    pq.write_table(table, table_path)
                    if split != "test":
                        tables.append(table) # keep table to create vocab
        
        # prepare vocab    
        vocab_path = os.path.join(self.data_dir, f"{self.dataset_name}_vocab.txt")
        if not os.path.exists(vocab_path):
            vocab = []
            table = pa.concat_tables(tables)
            qsts = table["question"].to_pylist()
            for qst in tqdm(qsts):
                tokenized = tokenize(qst)
                for word in tokenized:
                    if word not in vocab:
                        vocab.append(word)
            with open(vocab_path, "w") as f:
                f.write("[PAD]\n")
                for v in vocab:
                    f.write(f"{v}\n")
                f.write("[CLS]\n")
                f.write("[SEP]\n")
                f.write("[MASK]\n")
                f.write("[UNK]\n")
    
    def _prepare_dictionaries(self):
        pass
    
    def setup(self, stage):
        super().setup(stage)