# InteractionGraphNet (IGN)
  InteractionGraphNet: a Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Prediction and Large-scale Structure-based Virtual Screening

![error](https://github.com/zjujdj/InteractionGraphNet/blob/master/workflow_new.jpg)



# IGN Training (a toy example)
```
nohup python -u ./codes/ign_train.py --gpuid 0 --epochs 5 --repetitions 3 --lr 0.0001 --l2 0.000001 --dropout 0.1 > ./log/toy_example.log 2>&1 &
```
# Model Prediction
```
# step 1
# mol file format conversion
python3 ./codes/mol2tosdf.py --mol2path ./examples/mol2_files --sdfpath ./examples/sdf_files --num_process 12

# step 2
# select residues using chimera
python3 ./codes/select_residues.py --proteinfile ./examples/protein_6exw.pdb --sdfpath ./examples/sdf_files --finalpath ./examples/ign_input --num_process 12

# step 3
# prediction
python3 ./codes/prediction.py --cpu True --num_process 12 --input_path  ./examples/ign_input
```
