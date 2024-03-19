
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform
import os
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate
from torch_geometric.data import DataLoader, Dataset
##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import process
##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import process
#from dataload import loader_setup, get_dataset
from DSDCrystal import DSDCrystal
import argparse  # Add this line to import argparse

from utils_train import trainer, train,model_setup,model_summary,evaluate,write_results
from data import loader_setup,get_dataset,StructureDataset,GetY

model_parameters = {  
       "out_dims":64,
        "d_model":128,
        "N":3,
        "heads":4
        ,"dim1": 64
        ,"dim2": 64
        ,"numb_embbeding":1
        ,"numb_EGAT":5
        ,"numb_GATGCN":1
        ,"pool": "global_add_pool"
        ,"pool_order": "early"
        ,"act": "silu"
        ,"model": "DSDCrystal"
        ,"dropout_rate": 0.0
        ,"epochs": 800
        ,"lr": 0.006
        ,"batch_size": 80
        ,"optimizer": "AdamW"
        ,"optimizer_args": {}
        ,"scheduler": "ReduceLROnPlateau"
        ,"scheduler_args": {"mode":"min", "factor":0.8, "patience":15, "min_lr":0.00001, "threshold":0.0002}}
training_parameters = { 
    "target_index": 0
    ,"loss": "mse_loss"       
    ,"verbosity": 1
}
job_parameters= { 
        "reprocess":"False"
        ,"job_name": "my_train_job"   
        ,"load_model": "False"
        ,"save_model": "True"
        ,"model_path": "my_model_shear2.pth"
        ,"write_output": "True"
        ,"parallel": "True"
}
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--data", type=str, required=True, help=" dataset")
    args = parser.parse_args()

    # Assuming 'data_elasticity.pt' is in the specified data directory
    database_path = os.path.join(args.data_dir, args.data)
    
    # Setup the dataset
    database = get_dataset(args.data_dir, args.data, 0)

    # Setup loaders
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        0.85,
        0.05,
        0.10,
        90,
        database,
        'cuda',
        42,
        0,
    )

    # Setup the model
    model = model_setup(
        'cuda',
        'DSDCrystal',
        model_parameters,
        database,
    )

    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    trainer(
            'cuda',
            0,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            train_loader,
            val_loader,
            train_sampler,
            model_parameters["epochs"],
            training_parameters["verbosity"],
            "my_model_temp.pth",)
    

    train_error = val_error = test_error = float("NaN")

    ##workaround to get training output in DDP mode
    ##outputs are slightly different, could be due to dropout or batchnorm?
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_parameters["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Get train error in eval mode
    train_error, train_out = evaluate(
        train_loader, model, training_parameters["loss"], rank, out=True
    )
    print("Train Error: {:.5f}".format(train_error))

    ##Get val error
    if val_loader != None:
        val_error, val_out = evaluate(
            val_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Val Error: {:.5f}".format(val_error))

    ##Get test error
    if test_loader != None:
        test_error, test_out = evaluate(
            test_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Test Error: {:.5f}".format(test_error))


    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "full_model": model,
        },
        job_parameters["model_path"],
    )

    ##Write outputs
    if job_parameters["write_output"] == "True":

        write_results(
            train_out, str(job_parameters["job_name"]) + "_train_outputs.csv"
        )
        if val_loader != None:
            write_results(
                val_out, str(job_parameters["job_name"]) + "_val_outputs.csv"
            )
        if test_loader != None:
            write_results(
                test_out, str(job_parameters["job_name"]) + "_test_outputs.csv"
            )


if __name__ == "__main__":
    main()
