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
