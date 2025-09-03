"""
Contains dedicated functions to train, test and record ML results.
"""
import torch, sys, time
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Iterable, List
from tqdm import tqdm

from oil.utils.tunable_error import EvolvingLoss

# Define training cycle
MEAN_LOSS_KEY = "mean-loss"
MAX_LOSS_KEY = "max-loss"
MIN_LOSS_KEY = "min-loss"
MEAN_TEST_KEY = "mean-test"
TRAIN_TIMES_KEY = "train-times"
AVE_TRAIN_TIME_KEY = "mean-train-time"
TOTAL_TRAIN_TIME_KEY = "total-train-time"
TEST_TIMES_KEY = "test-time"
TOTAL_TIME_KEY = "time"
MODEL_KEY = "model"
OPTIMIZER_KEY = "optimizer"
EPOCHS_KEY = "n_epochs"
CRITERION_KEY = "criterion"
TEST_CRITERIA_KEY = "test-criteria"
TEST_RESULTS_KEY = "test_results"
def train_model(model:nn.Module, criterion:Callable, optimizer:torch.optim.Optimizer,
                train_dataloader:DataLoader, test_dataloader:DataLoader|None = None, n_epochs:int = 20,
                show_pbar:bool = True, test_criteria:Iterable[Callable]|dict|None = None, do_tests:bool = True)->dict:
    train_info = {
        MEAN_LOSS_KEY:[],
        MAX_LOSS_KEY:[],
        MIN_LOSS_KEY:[],
        MEAN_TEST_KEY:{},
        TRAIN_TIMES_KEY:[],
        TOTAL_TRAIN_TIME_KEY:0.0,
        TEST_TIMES_KEY:{},
        MODEL_KEY:str(model),
        EPOCHS_KEY:n_epochs,
        CRITERION_KEY:str(criterion),
        TEST_CRITERIA_KEY:[],
        OPTIMIZER_KEY:str(optimizer)
    }

    train_batches = len(train_dataloader)
    if (test_criteria is not None and test_dataloader is None) or do_tests:
        test_dataloader = train_dataloader
    if test_dataloader is not None:
        test_batches = len(test_dataloader)
        if test_criteria is None:
            test_criteria = [criterion]
    else:
        test_batches = 0

    if test_criteria is not None and test_dataloader is not None:
        if isinstance(test_criteria,dict):
            test_keys = list(test_criteria.keys())
            test_criteria_fns = list(test_criteria.values())
        else:
            test_criteria_fns = list(test_criteria)
            test_keys = [str(c) for c in test_criteria]

        assert test_criteria_fns
        assert len(test_criteria_fns) != 0 and len(test_keys) != 0
            
        for key in test_keys:
            train_info[MEAN_TEST_KEY][key]=[]
            train_info[TEST_TIMES_KEY][key]=[]
        train_info[TEST_CRITERIA_KEY] = test_keys
    else:
        test_criteria_fns = None
        test_keys = None

    pbar:None|tqdm = None
    if show_pbar:
        pbar = tqdm(total=n_epochs, desc="Training Model")

    
    T_START = time.perf_counter()

    for epoch_i in range(n_epochs):
        mean_loss = 0.0
        max_loss = 0.0
        min_loss = sys.float_info.max
        _t = time.perf_counter()

        # Train cycle
        for _input, _target in train_dataloader:

            # Zero out the gradients to avoid accumulation
            optimizer.zero_grad()

            # Update the criterion iff it is evolving
            if isinstance(criterion,EvolvingLoss):
                criterion.update(epoch_i)

            # Train on batch
            _output = model(_input)
            loss = criterion(_output, _target)
            loss.backward()
            optimizer.step()

            # Update loss records
            loss_item = loss.item()
            mean_loss += loss.item()
            if abs(loss_item) > abs(max_loss):
                max_loss = loss_item
            if abs(loss_item) < abs(min_loss):
                min_loss = loss_item
        
        # Divide mean train loss by number of batches
        mean_loss/=train_batches
        train_info[MAX_LOSS_KEY].append(max_loss)
        train_info[MEAN_LOSS_KEY].append(mean_loss)
        train_info[MIN_LOSS_KEY].append(min_loss)
        _t = time.perf_counter() - _t
        train_info[TRAIN_TIMES_KEY].append(_t)
        train_info[TOTAL_TRAIN_TIME_KEY] += _t


        # Run tests if applicable
        if test_dataloader is not None and test_criteria is not None:
            for key, test_criterion in zip(test_keys, test_criteria_fns): # type: ignore
                mean_test = 0
                if test_criterion is None:
                    test_criterion = criterion
                _t = time.perf_counter()
                for _input, _target in test_dataloader:
                    _output = model(_input)
                    error = test_criterion(_output, _target)
                    if isinstance(error, torch.Tensor):
                        if error.numel() == 1:
                            error = error.item()
                        else:
                            error = error.detach().numpy()
                    mean_test += error
                if test_batches > 0:
                    mean_test/=test_batches
                _t = time.perf_counter() - _t
                train_info[MEAN_TEST_KEY][key].append(mean_test)
                train_info[TEST_TIMES_KEY][key].append(_t)

        if show_pbar and pbar is not None:
            pbar.update()


    if show_pbar and pbar is not None:
        pbar.close()
        pbar = None

    # Record and format the time passed 
    t_passed = time.perf_counter()-T_START
    min,sec=divmod(int(t_passed),60)
    hour,min=divmod(min,60)
    time_str = f"{hour:02d}:{min:02d}:{sec:02d}"
    if show_pbar:
        print(f"Training complete in {time_str}.\nMean loss:\t{train_info[MEAN_LOSS_KEY][-1]}")
        if test_criteria_fns is not None and isinstance(test_criteria_fns, list) and len(test_criteria_fns) > 0:
            assert test_keys is not None and isinstance(test_keys, list)
            print(f"Test accuracy:\t{train_info[MEAN_TEST_KEY][test_keys[0]][-1]} ({test_keys[0]})")

    train_info[TOTAL_TRAIN_TIME_KEY] = t_passed
    train_info[AVE_TRAIN_TIME_KEY] = train_info[TOTAL_TRAIN_TIME_KEY]/n_epochs
    return train_info

def train_batch(models:dict, criterion:Callable, optimizer:Iterable[torch.optim.Optimizer],
                train_dataloader:DataLoader, test_dataloader:DataLoader|None = None, n_epochs:int = 20,
                show_pbar:bool = True, test_criteria:Iterable[Callable]|dict|None = None, do_tests:bool = True):
    train_infos = {}
    for key, model, optim in zip(models.keys(),models.values(),optimizer):
        if show_pbar:
            print(f" === {key} ===")
        train_infos[key] = train_model(model,
                                       criterion=criterion,
                                       optimizer=optim,
                                       train_dataloader=train_dataloader,
                                       test_dataloader=test_dataloader,
                                       n_epochs=n_epochs,
                                       show_pbar=show_pbar,
                                       test_criteria=test_criteria,
                                       do_tests=do_tests)
    return train_infos

def evaluate_model(model:nn.Module, test_dataloader, test_criteria:Iterable|dict):
    if isinstance(test_criteria,dict):
        test_criteria_fns = test_criteria.values()
        test_criteria_keys = test_criteria.keys()
    else:
        test_criteria_fns=test_criteria
        test_criteria_keys = [str(c) for c in test_criteria]

    test_batches = len(test_dataloader)
    test_results = {}
    for key, criterion in zip(test_criteria_keys, test_criteria_fns):
        e = 0
        for x,t in test_dataloader:
            e += criterion(model(x), t)

        if test_batches > 0:
            e /= test_batches
        test_results[key] = e

    return test_results


def evaluate_batch(models:dict, test_dataloader, test_criteria:Iterable|dict):
    test_results = {}
    for key, model in models.items():
        test_results[key] = evaluate_model(model, test_dataloader, test_criteria)
    return test_results

# def test_model(model:nn.Module, test_criterion:Callable|Iterable[Callable], test_dataloader:DataLoader|None = None, n_iters:int = 20,
#                 show_pbar:bool = True)->dict:
#     test_info={
#         TEST_TIMES_KEY:[],
#     }
