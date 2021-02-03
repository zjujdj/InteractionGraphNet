# InteractionGraphNet (IGN)
  InteractionGraphNet: a Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Prediction and Large-scale Structure-based Virtual Screening

![error](https://github.com/zjujdj/InteractionGraphNet/blob/master/workflow_new.jpg)


# Environment
```
# software
chimera v1.10.1 (https://www.cgl.ucsf.edu/chimera/cgi-bin/secure/chimera-get.py?file=linux_x86_64/chimera-1.10.1-linux_x86_64.bin)
openbabel (https://open-babel.readthedocs.io/en/latest/Installation/install.html#compiling-open-babel)

# python
python 3.6.5

# environment to reproduce
conda create --prefix xxx --file ./env/ign_needs_conda.txt
pip install -r ./env/ign_needs_pip.txt
```


# IGN Training (A toy example)
```
nohup python -u ./codes/ign_train.py --gpuid 0 --epochs 5 --repetitions 3 --lr 0.0001 --l2 0.000001 --dropout 0.1 > ./log/toy_example.log 2>&1 &
```
We added only about data of 200 toy samples in the data folder to explain how to train IGN model. Each sample is saved in a pickle file and it consists of two rdkit objects of a ligand and protein pocket prepared by chimera software. 


# Binding Affinity Prediction 
We use the well-trained IGN model to predict the binding affinity of complexes generated from docking program

```
# step 1
# mol file format conversion
# the mol2 files in ./examples/mol2_files folder are the conformers generated from docking program
python3 ./codes/mol2tosdf.py --mol2path ./examples/mol2_files --sdfpath ./examples/sdf_files --num_process 12

# step 2
# select residues using chimera for each ligand/protein pair
python3 ./codes/select_residues.py --proteinfile ./examples/protein_6exw.pdb --sdfpath ./examples/sdf_files --finalpath ./examples/ign_input --num_process 12

# step 3
# prediction
python3 ./codes/prediction.py --cpu True --num_process 12 --input_path  ./examples/ign_input
```
