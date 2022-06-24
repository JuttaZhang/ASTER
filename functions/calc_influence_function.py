#! /usr/bin/env python3

import torch
import time
import datetime
import numpy as np
import copy
import logging
import sys

from tqdm import tqdm
from pathlib import Path
from functions.influence_function import s_test, grad_z

def calc_s_test(model, test_loader, train_loader, save=False, gpu=-1,
                damp=0.01, scale=25, recursion_depth=5000, r=1, start=0):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.
    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0
    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    iterator = tqdm(range(start, len(test_loader.dataset)), desc="Calc. z_test (s_test):")
    for i in range(1):
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,
                                        gpu, damp, scale, recursion_depth, r)

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec,
                save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test"))
        else:
            s_tests.append(s_test_vec)

    return s_tests, save



def calc_s_test_single(model, z_test, t_test, train_loader, gpu=-1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    test_loss = 0
    for i in tqdm(range(r), desc="Averaging r-times: "):
        temp, test_l = s_test(z_test, t_test, model, train_loader,
               gpu=gpu, damp=damp, scale=scale,
               recursion_depth=recursion_depth)
        s_test_vec_list.append(temp)
        test_loss += test_l
    loss = test_loss / r
    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################
    print('len of s_test_vec_list:{}'.format(len(s_test_vec_list)))

    s_test_vec = s_test_vec_list[0]
    for i in range(len(s_test_vec_list)):
        s_test_vec = [i + j for i, j in zip(s_test_vec, s_test_vec_list[0])]
    s_test_vec = [i / r for i in s_test_vec]

    print('len of s_test_vec:{}'.format(len(s_test_vec)))

    return s_test_vec, loss

def get_dataset_sample_ids_per_class(class_id, num_samples, test_loader,
                                     start_index=0):
    """Gets the first num_samples from class class_id starting from
    start_index. Returns a list with the indicies which can be passed to
    test_loader.dataset[X] to retreive the actual data.
    Arguments:
        class_id: int, name or id of the class label
        num_samples: int, number of samples per class to process
        test_loader: DataLoader, can load the test dataset.
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.
    Returns:
        sample_list: list of int, contains indicies of the relevant samples"""
    sample_list = []
    img_count = 0
    for i in range(len(test_loader.dataset)):
        _, t = test_loader.dataset[i]
        if class_id == t:
            img_count += 1
            if (img_count > start_index) and \
                    (img_count <= start_index + num_samples):
                sample_list.append(i)
            elif img_count > start_index + num_samples:
                break

    return sample_list

def get_dataset_sample_ids(num_samples, test_loader, num_classes=None,
                           start_index=0):
    """Gets the first num_sample indices of all classes starting from
    start_index per class. Returns a list and a dict containing the indicies.
    Arguments:
        num_samples: int, number of samples of each class to return
        test_loader: DataLoader, can load the test dataset
        num_classes: int, number of classes contained in the dataset
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.
    Returns:
        sample_dict: dict, containing dict[class] = list_of_indices
        sample_list: list, containing a continious list of indices"""
    sample_dict = {}
    sample_list = []
    if not num_classes:
        num_classes = len(np.unique(test_loader.dataset.targets))
    for i in range(num_classes):
        sample_dict[str(i)] = get_dataset_sample_ids_per_class(
            i, num_samples, test_loader, start_index)
        # Append the new list on the same level as the old list
        # Avoids having a list of lists
        sample_list[len(sample_list):len(sample_list)] = sample_dict[str(i)]
    return sample_dict, sample_list

def calc_img_wise(model, train_loader, test_loader, test_sample_num=1, test_start_index=0,
                  num_classes=10, gpu=0, damp=0.01, scale=25, recursion_depth=1, r=1, save=False,):
    """Calculates the influence function one test point at a time. Calculates
    the `s_test` and `grad_z` values on the fly and discards them afterwards.
    Arguments:
        config: dict, contains the configuration from cli params"""

    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    test_loss = 0
    if test_sample_num and test_start_index is not False:
        test_dataset_iter_len = test_sample_num * num_classes
        _, sample_list = get_dataset_sample_ids(test_sample_num, test_loader,
                                                num_classes,
                                                test_start_index)
    else:
        test_dataset_iter_len = len(test_loader.dataset)

    for j in tqdm(range(test_dataset_iter_len), desc="Test samples processed"):
        if test_sample_num and test_start_index:
            if j >= len(sample_list):
                logging.warn("ERROR: the test sample id is out of index of the"
                             " defined test set. Jumping to next test sample.")
                next
            i = sample_list[j]
        else:
            i = j
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec, test_l = calc_s_test_single(model, z_test, t_test, train_loader,
                                        gpu, damp, scale, recursion_depth, r)

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec,
                save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test"))
        else:
            s_tests.append(s_test_vec)
        test_loss += test_l
    loss = test_loss/test_dataset_iter_len
    return s_tests, save, loss




def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'seed': 42,
        'gpu': 0,
        'dataset': 'CIFAR10',
        'num_classes': 10,
        'test_sample_num': 1,
        'test_start_index': 0,
        'recursion_depth': 1,
        'r_averaging': 1,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }
    return config

