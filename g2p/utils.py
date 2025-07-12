UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def validate_datalist(data: list[str]) -> None:
    data = list(data)
    for i, symbol in enumerate(special_symbols):
        assert data[i] == symbol, f"Element at index {i} must be {symbol}, but got {data[i]}"
    assert len(data) == len(set(data)), "Data must not contain duplicates"
    assert len(data) > 4, "Phoneme/Grapheme list must contain at least 5 elements."
