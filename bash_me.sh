#!/bin/bash

# def setup_args():
#     parser = argparse.ArgumentParser(description='Train a neural network for donor prediction.')
#     parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
#     parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
#     parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
#     parser.add_argument('--layers', nargs='+', type=int, default=[100, 200], help='List of sizes for each hidden layer.')
#     parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation.')
#     parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate.')
#     return parser.parse_args()

# NO
# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python fundraising_cross_val_CA.py \
# --patience 10 \
# --lr .0005 \
# --epochs 10 \
# --batch_size 8 \
# --layers 100 100 \
# --folds 10 \
# --dropout .2 \
# --i = 0' > nn_model_results.jsonl &

# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python fundraising_cross_val_CA.py \
# --patience 10 \
# --lr .0005 \
# --epochs 10 \
# --batch_size 8 \
# --layers 100 100 \
# --folds 10 \
# --dropout .2 \
# --i = 1' >> nn_model_results.jsonl &


#!/bin/bash

# Start with a clean output file

# CHECK THIS OUT

> nn_model_results.jsonl

for i in {0..799}; do
    patience=$((5 + i % 16))  # Vary patience from 5 to 20
    lr=$(echo "scale=5; 0.0005 + $i * 0.0000125" | bc)  # Vary lr from 0.0005 to 0.01
    epochs=$((10 + i % 31))  # Vary epochs from 10 to 40
    batch_size=$((8 + i % 17))  # Vary batch_size from 8 to 24
    layers="100 $((100 + i % 201))"  # Vary second layer size from 100 to 300
    folds=5  # Keep folds constant at 5
    dropout=$(echo "scale=2; 0.05 + $i * 0.000225" | bc)  # Vary dropout from 0.05 to 0.5

    CUDA_VISIBLE_DEVICES=0 nohup python fundraising_cross_val_CA.py \
    --patience $patience \
    --lr $lr \
    --epochs $epochs \
    --batch_size $batch_size \
    --layers $layers \
    --folds $folds \
    --dropout $dropout \
    --i=$i >> nn_model_results.jsonl
done
