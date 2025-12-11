##Installation##
conda env update -f environment.yml
conda activate mort24

This will install the required libraries and create a conda environment.

##Training##
Transformer
cd mort24
python run.py 
python run.py -help 

DANN
python train_dann.py \
    --source data_chop/eicu \
    --target data_chop/mimic \
    --epochs 50 \
    --batch-size 32

python train_dann.py \
    --source <path>          # Source domain data directory
    --target <path>          # Target domain data directory
    --epochs 100             # Number of training epochs
    --batch-size 32          # Batch size
    --lr 1e-4                # Learning rate
    --hidden 128             # Transformer hidden dimension
    --heads 4                # Number of attention heads
    --depth 2                # Number of transformer layers
    --cpu                    # Force CPU (otherwise uses GPU if available)
    --verbose                # Enable debug logging

##Architecture##
┌─────────────────────────────────────────────────────────────────┐
│                        DANN-Transformer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (batch, seq_len, features)                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────┐                │
│  │         Feature Extractor                   │                │
│  │  ┌─────────────────────────────────────┐    │                │
│  │  │ Input Embedding (Linear)            │    │                │
│  │  │ Positional Encoding                 │    │                │
│  │  │ Transformer Blocks × depth          │    │                │
│  │  └─────────────────────────────────────┘    │                │
│  └─────────────────────────────────────────────┘                │
│       │                                                         │
│       ▼                                                         │
│  Features (batch, seq_len, hidden)                              │
│       │                                                         │
│       ├───────────────────┬─────────────────────┐               │
│       │                   │                     │               │
│       ▼                   ▼                     │               │
│  ┌──────────┐    ┌────────────────┐             │               │
│  │  Task    │    │    Gradient    │             │               │
│  │Classifier│    │    Reversal    │             │               │
│  │ (Linear) │    │     Layer      │             │               │
│  └──────────┘    └────────────────┘             │               │
│       │                   │                     │               │
│       ▼                   ▼                     │               │
│  Task Output      ┌────────────────┐            │               │
│  (mortality)      │    Domain      │            │               │
│                   │ Discriminator  │            │               │
│                   └────────────────┘            │               │
│                          │                      │               │
│                          ▼                      │               │
│                   Domain Output                 │               │
│                   (source/target)               │               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘