import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .utils import BOS_IDX, UNK_IDX

class Encoder(nn.Module):
    def __init__(self,
                graphemes: list[str],
                d_model: int,
                d_hidden: int,
                num_layers: int,
                dropout: float) -> None:
        super().__init__()
        
        # list of graphemes
        self.graphemes = graphemes

        # model dimension size
        self.d_model = d_model

        # number of hidden layers
        self.d_hidden = d_hidden

        # number of stacked lstm layers
        self.num_layers = num_layers

        # dropout
        self.dropout = dropout

        # token embeddings
        self.embedding = nn.Embedding(len(graphemes), d_model)

        # bidirectional lstm layer
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_hidden // 2, # bidirectional
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        
        # linear projection layer
        self.fc = nn.Linear(d_hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # input tensor x: (batch_size, seq_len) - indices of graphemes

        # convert indicies to dense vectors
        # x: (batch_size, seq_len, d_model)
        x = self.embedding(x)

        # optimize lstm params
        self.lstm.flatten_parameters()

        # process through bidirectional lstm
        # x: (batch_size, seq_len, d_hidden)
        x, _ = self.lstm(x)

        # apply linear projection
        # x: (batch_size, seq_len, d_model)
        return self.fc(x)
    
class Decoder(nn.Module):
    def __init__(self,
                phonemes: list[str],
                d_model: int,
                d_hidden: int,
                num_layers: int,
                dropout: float) -> None:
        super().__init__()

        # list of phonemes
        self.phonemes = phonemes

        # model dimension size
        self.d_model = d_model

        # number of hidden layers
        self.d_hidden = d_hidden

        # number of stacked lstm layers
        self.num_layers = num_layers

        # dropout
        self.dropout = dropout

        # token embeddings
        self.embedding = nn.Embedding(len(phonemes), d_model)

        # lstm layer
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        
        # linear projection layer
        self.fc = nn.Linear(d_hidden, d_model)

        # joint layer
        self.joint = Joint(d_model, d_hidden, len(phonemes))

    def forward(self, input: Tensor, memory: Tensor, hidden_state=None, cell_state=None) -> tuple[Tensor, Tensor, Tensor]:
        # input tensor input: (batch_size, seq_len) - indices of phonemes

        # convert indices to dense vectors
        # input: (batch_size, seq_len, d_model)
        x = self.embedding(input)

        # optimize lstm params
        self.lstm.flatten_parameters()

        # process through lstm, if hidden_state and cell_state are provided, use them
        # x: (batch_size, seq_len, d_hidden)
        x, (hidden_state, cell_state) = self.lstm(x, None if hidden_state is None else (hidden_state, cell_state))

        # apply linear projection
        # x: (batch_size, seq_len, d_model)
        x = self.fc(x)

        # join the encoder and decoder states
        return self.joint(memory, x), hidden_state, cell_state
    
    def encode_full(self, input: Tensor) -> Tensor:
        # input tensor: (batch_size, seq_len) - indices of phonemes

        # convert indices to dense vectors
        # input: (batch_size, seq_len, d_model)
        x = self.embedding(input)

        # optimize lstm params
        self.lstm.flatten_parameters()

        # process through lstm
        # x: (batch_size, seq_len, d_hidden)
        x, _ = self.lstm(x)

        # apply linear projection
        # x: (batch_size, seq_len, d_model)
        return self.fc(x)
    
    def step(self, input: Tensor, memory: Tensor, time_index: Tensor, hidden_state: Tensor, cell_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # input tensor: (batch_size, 1) - indices of phonemes
        # memory tensor: (batch_size, seq_len, d_model) - encoder output
        # time_index tensor: (batch_size, 1) - current time step index

        # embed last token
        # input: (batch_size, 1, d_model)
        x = self.embedding(input[:, -1:])

        # select memory for current time step
        mem = memory[:, time_index[0]].unsqueeze(1)

        # lstm forward pass
        # x: (batch_size, 1, d_hidden)
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))

        # apply linear projection
        # x: (batch_size, 1, d_model)
        x = self.fc(x)

        # combine encoder and decoder states
        # x: (batch_size, 1, len(phonemes))
        x = self.joint(mem, x)

        # apply softmax and argmax
        x = F.softmax(x, dim=-1)
        x = torch.argmax(x, dim=-1).reshape(1, -1)

        # change memory format
        x = x.int()

        return x, hidden_state, cell_state

class Joint(nn.Module):
    def __init__(self,
                d_model: int,
                d_hidden: int,
                d_output: int) -> None:
        super(Joint, self).__init__()

        self.forward_layer = nn.Linear(d_model * 2, d_hidden, bias=True)
        self.tanh_layer = nn.Tanh()
        self.projection_layer = nn.Linear(d_hidden, d_output, bias=True)

    def forward(self, encoder_state: Tensor, decoder_state: Tensor) -> Tensor:
        # check dimensions of input tensors
        assert encoder_state.dim() == 3, "Encoder state must be of shape (batch_size, seq_len, d_model)"
        assert decoder_state.dim() == 3, "Decoder state must be of shape (batch_size, seq_len, d_model)"

        # get the lengths of the encoder and decoder states
        encoder_length = encoder_state.size(1)
        decoder_length = decoder_state.size(1)

        # expand tensors
        encoder_state = encoder_state.unsqueeze(2).expand(-1, -1, decoder_length, -1)
        decoder_state = decoder_state.unsqueeze(1).expand(-1, encoder_length, -1, -1)

        # concatenate encoder and decoder states
        joint_state = torch.cat((encoder_state, decoder_state), dim=-1)  # (batch_size, encoder_length, decoder_length, d_model * 2)

        output = self.forward_layer(joint_state)  # (batch_size, encoder_length, decoder_length, d_hidden)
        output = self.tanh_layer(output)  # (batch_size, encoder_length, decoder_length, d_hidden)
        output = self.projection_layer(output)  # (batch_size, encoder_length, decoder_length, d_output)

        return output  # (batch_size, encoder_length, decoder_length, d_output)

class LstmG2P(nn.Module):
    def __init__(self,
                 max_len: int,
                 encoder: Encoder,
                 decoder: Decoder) -> None:
        super().__init__()

        self.max_len = max_len
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, graphemes: Tensor, phonemes: Tensor) -> Tensor:
        # graphemes tensor: (batch_size, seq_len) - indices of graphemes
        # phonemes tensor: (batch_size, seq_len) - indices of phonemes

        # encode graphemes
        # memory: (batch_size, seq_len, d_model)
        memory = self.encoder(graphemes)

        # encode phonemes
        # x: (batch_size, seq_len, d_model)
        x, _, _ = self.decoder(phonemes, memory)

        return x
    
    def predict(self, src: Tensor) -> Tensor:
        # src tensor: (batch_size, seq_len) - indices of graphemes
        
        with torch.no_grad():
            # model mode
            self.eval()

            # select device
            device = next(self.parameters()).device

            # move tensors to device
            graphemes = src.to(device)
            phonemes = torch.tensor([[BOS_IDX]]).to(device)

            memory = self.encoder(graphemes)
            graphemes_length = graphemes.shape[-1]

            # initialize time index and hidden states
            # time_index: (1) - current time step index
            # hidden_state: (num_layers, 1, d_hidden) - initial hidden state
            # cell_state: (num_layers, 1, d_hidden) - initial cell state
            time_index = torch.zeros((1)).int().to(device)
            hidden_state = torch.zeros((self.decoder.num_layers, 1, self.decoder.d_hidden)).to(device)
            cell_state = torch.zeros((self.decoder.num_layers, 1, self.decoder.d_hidden)).to(device)

            while time_index < graphemes_length and phonemes.shape[1] < self.max_len:
                prediction, new_hidden_state, new_cell_state = self.decoder.step(phonemes, memory, time_index, hidden_state, cell_state)
                if prediction.item() != BOS_IDX:
                    phonemes = torch.concat([phonemes, prediction], dim=-1)
                    hidden_state = new_hidden_state
                    cell_state = new_cell_state
                else:
                    time_index[0] += 1

        return phonemes
    
    def predict_str(self, word: str) -> list[str]:
        self.grapheme_indexes = {grapheme: i for i, grapheme in enumerate(self.encoder.graphemes)}
        graphemes = torch.tensor([[self.grapheme_indexes.get(grapheme.lower(), UNK_IDX) for grapheme in word]])

        phonemes = self.predict(graphemes)
        phonemes = phonemes.cpu().int().numpy().tolist()

        return [self.decoder.phonemes[i] for i in phonemes[0][1:]]

class SimplifiedLstmG2P(nn.Module):
    def __init__(self,
                max_len: int,
                encoder: Encoder,
                decoder: Decoder) -> None:
        super().__init__()

        self.max_len = max_len
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, graphemes: Tensor, phonemes: Tensor, time_index : Tensor) -> Tensor:
        # graphemes tensor: (batch_size, seq_len) - indices of graphemes
        # phonemes tensor: (batch_size, seq_len) - indices of phonemes
        # time_index tensor: (batch_size, 1) - current time step index

        # encode graphemes
        # memory: (batch_size, seq_len, d_model)
        memory = self.encoder(graphemes)

        # select memory for current time step
        memory = memory[:, time_index[0]].unsqueeze(1)

        # encode phonemes
        x = self.decoder.encode_full(phonemes)

        # select last decoder output
        x = x[:, -1].unsqueeze(1)

        # combine encoder and decoder states
        # x: (batch_size, 1, len(phonemes))
        x = self.decoder.joint(memory, x)

        # apply softmax and argmax
        x = F.softmax(x[0, 0], dim=-1)
        prediction = torch.argmax(x, dim=-1).int()

        return prediction
    
    def predict(self, graphemes: Tensor) -> Tensor:
        # graphemes tensor: (batch_size, seq_len) - indices of graphemes

        phonemes = torch.tensor([2], dtype=torch.int32)
        time_index = torch.tensor([0], dtype=torch.int32)

        while time_index[0] < graphemes.shape[1] and phonemes.shape[0] < self.max_len:
            prediction = self.forward(graphemes, phonemes.unsqueeze(0), time_index)

            if prediction.item() != BOS_IDX:
                phonemes = torch.concat([phonemes, prediction], dim=0)
            else:
                time_index[0] += 1

        return phonemes.unsqueeze(0)
    
    def export(self, path: str) -> None:
        self.eval()
        src = torch.zeros((1, 8)).int()
        tgt = torch.zeros((1, 6)).int()
        t = torch.tensor([0]).int()
        torch.onnx.export(
            self, (src, tgt, t), path,
            input_names=['src', 'tgt', 't'],
            output_names=['pred'],
            dynamic_axes={
                'src': {1: 'T'},
                'tgt': {1: 'U'},
            },
            opset_version=11)