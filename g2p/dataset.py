import torch
from torch.utils.data import Dataset

from .utils import validate_datalist, UNK_IDX


class TSVDataset(Dataset):
    def __init__(self, dict_path: str, graphemes: list[str], phonemes: list[str]) -> None:
        self.graphemes = graphemes
        self.phonemes = phonemes

        self.grapheme_indexes = {symbol: i for i, symbol in enumerate(self.graphemes)}
        self.phoneme_indexes = {symbol: i for i, symbol in enumerate(self.phonemes)}

        validate_datalist(self.graphemes)
        validate_datalist(self.phonemes)

        self.entries = self.load_dict(dict_path)

    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            # return a list of (word_tensor, phoneme_tensor) for each index (batch)
            return [self.__getitem__(i) for i in idx]

        word, phonemes = self.entries[idx]

        word_tensor = torch.tensor([self.grapheme_indexes.get(grapheme, UNK_IDX) for grapheme in word], dtype=torch.int32)
        phoneme_tensor = torch.tensor([self.phoneme_indexes.get(phoneme, UNK_IDX) for phoneme in phonemes], dtype=torch.int32)

        return word_tensor, phoneme_tensor

    def load_dict(self, dict_path: str) -> list[tuple[str, list[str]]]:
        entries = []
        ignored_graphemes = set()
        ignored_phonemes = set()

        with open(dict_path, 'r', encoding='utf-8') as dict_file:
            for line in dict_file.readlines():
                line = line.strip()

                # [0] - Word
                # [1] - Phonemes (space delimited)
                entry = line.split('\t')
                assert len(entry) == 2, f"Invalid entry in dictionary: {line}"

                word = entry[0]
                phonemes = entry[1].split(' ')

                entries.append((word, phonemes))

        return entries
    
    def metrics(self) -> dict:
        return {
            'num_entries': len(self.entries),
            'num_graphemes': len(self.graphemes),
            'num_phonemes': len(self.phonemes),
            'graphemes': self.graphemes,
            'phonemes': self.phonemes
        }