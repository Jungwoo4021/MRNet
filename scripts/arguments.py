import os
import torch
import itertools

def get_args():
    """
    Returns
        system_args (dict): path, log setting
        experiment_args (dict): hyper-parameters
        args (dict): system_args + experiment_args
    """
    system_args = {
        # expeirment info
        'project'       : 'MLFA',
        'name'          : '',
        'tags'          : ['FMA', 'proposed_model'],
        'description'   : '',

        # log
        'path_log'      : '/results',
        'neptune_user'  : '',
        'neptune_token' : '',
        'wandb_group'   : '',
        'wandb_entity'  : '',

        # dataset
        'path_FMA'      : '/data/fma_dataset',
        'FMA_size'      : 'small',

        # others
        'num_workers'   : 4,
        'usable_gpu'    : None,
    }

    experiment_args = {
        # huggingface model
        'huggingface_url'    : 'microsoft/wavlm-base-plus',
        
        # experiment
        'epoch'             : 1000,
        'batch_size'        : 48,
        'rand_seed'		    : 1,
        
        # data processing
        'sample_rate'       : 16000,
        'num_train_frames'  : 16000 * 6,
        'p_gaussian_noise'  : 0.7,
        'gaussian_mean'     : [-0.03, 0.03],
        'gaussian_std'      : [0.05, 0.1],
        
        # model
        'C'                 : 512,
        'embed_size'        : 128,
        'hidden_size'       : 768,
        'hidden_num'        : 13,
        
        # learning rate
        'lr'                : 1e-3,
        'lr_min'            : 5e-4,
        'weight_decay'      : 1e-4,
        'T_0'               : 1000,
        'T_mult'            : 2,
    }

    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))
    args['path_log'] = os.path.join(args['path_log'], args['project'], args['name'])

    return args, system_args, experiment_args