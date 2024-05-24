import dataclasses
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
from dataclasses import dataclass


@dataclass
class Paths:
    valid_path = "latex_data/latex_data/valid.lst"
    train_path = "latex_data/latex_data/train.lst"
    test_path = "latex_data/latex_data/test.lst"
    image_path = "latex_data/latex_data/images_processed"
    formulas_list_path = "latex_data/latex_data/formulas.norm.lst"


class MathFormulaDataset(Dataset):
    def __init__(self,
                 train_path,tokenizer,
                 image_path="latex_data/latex_data/images_processed"
                 ,formulas_list_path="latex_data/latex_data/formulas.norm.lst"
                 ):
        self.tokenizer = tokenizer
        self.image_path = image_path
        self.formulas_list = self.load_formulas(formulas_list_path)
        self.data = self.load_data(train_path)
        self.transform = transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
        ])
    def load_formulas(self, formulas_list_path):
        formulas_list = []
        with open(formulas_list_path, "r") as f:
            for line in f:
                formulas_list.append(line.strip())

        return formulas_list
    def load_data(self, train_path):
        data = {"image": [], "id_formula": []}
        with open(train_path, "r") as f:
            for line in f:
                s = line.split()
                data["image"].append(s[0])
                data["id_formula"].append(int(s[1]) - 1)
        return data

    def __len__(self):
        return len(self.data["image"])

    def __getitem__(self, idx):
        image_name = self.data["image"][idx]
        image = Image.open(os.path.join(self.image_path, image_name))
        formula_id = self.data["id_formula"][idx]
        formula = self.formulas_list[formula_id]
        formula = '<SOS> ' + formula + ' <EOS>'
        if self.transform:
            image = self.transform(image)

        # Convert formula to tokens using the tokenizer
        tokens = self.tokenizer.formula_to_tokens(formula)
        # Convert tokens to tensor
        formula_tensor = torch.tensor(tokens)

        return image, formula_tensor, len(tokens)

    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, targets, captions = zip(*data)
        images = torch.stack([self.transform(image) for image in images], 0)
        lengths = [len(tar) for tar in targets]
        _targets = torch.zeros(len(captions), max(lengths)).long()
        for i, tar in enumerate(targets):
            end = lengths[i]
            _targets[i, :end] = tar[:end]
        # Ensure captions are returned as a tensor
        captions = torch.stack(captions)
        return images, _targets, lengths

def get_dataloader(path,tokenizer,batch_size=8):
    dataset = MathFormulaDataset(path,tokenizer=tokenizer)
    dl = DataLoader(dataset,batch_size,num_workers=2,pin_memory=True,shuffle=True)
    return dl
def get_vocab(path="latex_data/latex_data/latex_vocab.txt"):
    with open(path, "r") as f:
        words = f.read().split()

    vocab = {}
    idx = 0
    for word in words:
        if word not in vocab:
            vocab[word] = idx
            idx += 1

    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    return vocab

class Tokenizer:
    # Takes as an input the vocab of your text
    def __init__(self, vocab):
        self.vocab = vocab
        self.reverse_vocab = self._reverse_dict(self.vocab)

    def formula_to_tokens(self, formula):
        encoding = []
        formula = formula.split()
        for ix in formula:
            encoding.append(self.vocab[f"{ix}"])
        return encoding

    def _reverse_dict(self, dict):
        return {v: k for k, v in dict.items()}

    def tokens_to_formula(self, tokens):
        decoding = []
        for ix in tokens:
            decoding.append(self.reverse_vocab[ix])
        return decoding

    def get_string(self, encoded_string):
        string = ''
        for i in encoded_string:
            string += ' ' + i
        string = string[1:]
        return string

    def vocab_size(self):
        return len(self.vocab)