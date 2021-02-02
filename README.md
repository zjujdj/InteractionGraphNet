# InteractionGraphNet
InteractionGraphNet: a Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Prediction and Large-scale Structure-based Virtual Screening


# ign training (toy example)
nohup python -u ./codes/ign_train.py --gpuid 0 --epochs 5 --repetitions 3 --lr 0.0001 --l2 0.000001 --dropout 0.1 > ./log/toy_example.log 2>&1 &
