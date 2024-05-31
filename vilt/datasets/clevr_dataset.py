from .base_dataset import BaseDataset


class ClevrDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        
        if split == "train":
            names = ["clevr_train"]
        elif split == "val":
            names = ["clevr_val"]
        elif split == "test":
            names = ["clevr_test"]
            
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="question",
            remove_duplicate=False,
        )
    
    def __getitem__(self, index):
        import pdb; pdb.set_trace()
