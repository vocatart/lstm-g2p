from g2p.trainer import Trainer
import torch
from g2p.dataset import TSVDataset
from g2p.model import LstmG2P, SimplifiedLstmG2P, Encoder, Decoder
from torchinfo import summary
import yaml
import os

def train(trainer: Trainer):
    trainer.train()

def dump_trainer_info(trainer) -> dict:
    dict = {
        'run_name': trainer.run_name,
        'run_datetime': trainer.run_datetime,
        'device': str(trainer.device),
        'loss_device': str(trainer.loss_device),
        'model': trainer.model.__class__.__name__,
        'exported_model_type': SimplifiedLstmG2P.__name__,
        'dataset': trainer.dataset.__class__.__name__,
        'batch_size': trainer.batch_size,
        'num_epochs': trainer.num_epochs,
        'learning_rate': trainer.learning_rate,
        'minimum_learnning_rate': trainer.minimum_learning_rate,
        'gamma': trainer.gamma,
        'grad_clip': trainer.grad_clip,
        'validation_divide_by': trainer.validation_divide_by,
        'dl_workers': trainer.dl_workers,
        'seed': trainer.seed,
        'max_len': trainer.model.max_len,
        'encoder': {
            'graphemes': trainer.model.encoder.graphemes,
            'd_model': trainer.model.encoder.d_model,
            'd_hidden': trainer.model.encoder.d_hidden,
            'num_layers': trainer.model.encoder.num_layers,
            'dropout': trainer.model.encoder.dropout
        },
        'decoder': {
            'phonemes': trainer.model.decoder.phonemes,
            'd_model': trainer.model.decoder.d_model,
            'd_hidden': trainer.model.decoder.d_hidden,
            'num_layers': trainer.model.decoder.num_layers,
            'dropout': trainer.model.decoder.dropout
        },
    }

    return dict

def export(trainer: Trainer):
    artifacts_dir = trainer.artifacts_dir
    
    statedict_path = os.path.join(artifacts_dir, 'statedict-best.pt')
    onnx_path = os.path.join(artifacts_dir, 'g2p.onnx')
    model_info_path = os.path.join(artifacts_dir, 'info.yaml')
    
    print(f"exporting ONNX and metadata to: {artifacts_dir}")
    
    print("exporting model to ONNX format...")
    trainer.model.load_state_dict(torch.load(statedict_path))
    trainer.model.cpu()
    trainer.model.eval()

    simplified = SimplifiedLstmG2P(
        trainer.model.max_len,
        trainer.model.encoder,
        trainer.model.decoder
    )

    # freeze
    for param in simplified.parameters():
        param.requires_grad = False

    summary(simplified, 
            input_size=[(1, 8), (1, 6), (1,)],  # input sizes are src, tgt, t
            dtypes=[torch.int32, torch.int32, torch.int32],
            device='cpu',
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            )

    print(f"\ncreating metadata file: {model_info_path}")

    model_info = dump_trainer_info(trainer)
    
    with open(model_info_path, 'w') as f:
        yaml.dump(model_info, f, default_flow_style=False)

    simplified.export(onnx_path)
    print(f"onnx model exported: {onnx_path}")
    
    return artifacts_dir

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    config_path = 'g2p/data/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"loaded configuration from: {config_path}")
    
    dict_path = config['data']['dict_path']
    
    # build vocab
    graphemes = set()
    phonemes = set()

    with open(dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                word = parts[0]
                phoneme_list = parts[1].split()
                graphemes.update(word.lower())
                phonemes.update(phoneme_list)

    # process vocab
    graphemes = ['<unk>', '<pad>', '<bos>', '<eos>'] + sorted(list(graphemes))
    phonemes = ['<unk>', '<pad>', '<bos>', '<eos>'] + sorted(list(phonemes))

    # create dataset
    dataset = TSVDataset(dict_path, graphemes, phonemes)

    model = LstmG2P(
        max_len=config['model']['max_len'],
        encoder=Encoder(
            graphemes=graphemes,
            d_model=config['encoder']['d_model'],
            d_hidden=config['encoder']['d_hidden'],
            num_layers=config['encoder']['num_layers'],
            dropout=config['encoder']['dropout']
        ),
        decoder=Decoder(
            phonemes=phonemes,
            d_model=config['decoder']['d_model'],
            d_hidden=config['decoder']['d_hidden'],
            num_layers=config['decoder']['num_layers'],
            dropout=config['decoder']['dropout']
        )
    )

    trainer = Trainer(
        run_name=config['training']['run_name'],
        device=device,
        loss_device=device,
        model=model,
        dataset=dataset,
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        minimum_learning_rate=config['training']['minimum_learning_rate'],
        gamma=config['training']['gamma'],
        grad_clip=config['training']['grad_clip'],
        validation_divide_by=config['training']['validation_divide_by'],
        dl_workers=config['training']['dl_workers'],
        seed=config['training']['seed']
    )

    print(dataset.metrics())

    summary(model, 
            input_size=[(config['training']['batch_size'], config['model']['max_len']), (config['training']['batch_size'], config['model']['max_len'])], 
            dtypes=[torch.long, torch.long],
            device=device,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            )

    train(trainer)

    export(trainer)

    artifacts_dir = trainer.artifacts_dir
    print(f"\nall artifacts exported to: {artifacts_dir}")
    print("contents:")
    for file in os.listdir(artifacts_dir):
        file_path = os.path.join(artifacts_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file} ({size:,} bytes)")

