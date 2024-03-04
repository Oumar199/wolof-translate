from wolof_translate import *
import warnings

def train(config: dict):
    
    # ---------------------------------------
    # add distribution if necessary (https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/pytorch_mnist/mnist.py)
    
    logger = config['logger']
    
    is_distributed = len(config['hosts']) > 1 and config['backend'] is not None
    
    use_cuda = config['num_gpus'] > 0
    
    config.update({"num_workers": 1, "pin_memory": True} if use_cuda else {})

    if not logger is None:
        
        logger.debug("Distributed training - {}".format(is_distributed))
        
        logger.debug("Number of gpus available - {}".format(config['num_gpus']))
        
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(config['hosts'])
        
        os.environ["WORLD_SIZE"] = str(world_size)
        
        host_rank = config['hosts'].index(config['current_host'])
        
        os.environ["RANK"] = str(host_rank)
        
        dist.init_process_group(backend=config['backend'], rank=host_rank, world_size=world_size)
        
        if not logger is None: logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                config['backend'], dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), config['num_gpus'])
        )
    # ---------------------------------------
    
    # split the data
    if config['include_split']: split_data(config['random_state'], config['data_directory'], config['data_file'])

    # recuperate the tokenizer
    tokenizer = T5TokenizerFast(config['tokenizer_path'])
    
    # Initialize the model name
    model_name = 't5-small'

    # import the model with its pre-trained weights
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # resize the token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # recuperate train and test set
    train_dataset, test_dataset = recuperate_datasets(config['char_p'],
                                                        config['word_p'], 50,
                                                        config['end_mark'], tokenizer, config['corpus_1'],
                                                        config['corpus_2'],
                                                        config['train_file'], config['test_file'])
    
    # initialize the evaluation object
    evaluation = TranslationEvaluation(tokenizer, train_dataset.decode)

    # let us initialize the trainer
    trainer = ModelRunner(model = model, version=config['version'], seed = 0, evaluation = evaluation, optimizer = Adafactor)

    #-------------------------------------
    # in the case when the linear learning rate scheduler with warmup is used
    
    # let us calculate the appropriate warmup steps (let us take a max epoch of 100)
    # length = len(train_dataset)

    # n_steps = length // config['batch_size']

    # num_steps = config['max_epoch'] * n_steps

    # warmup_steps = (config['max_epoch'] * n_steps) * config['warmup_ratio']

    # Initialize the scheduler parameters
    # scheduler_args = {'num_warmup_steps': warmup_steps, 'num_training_steps': num_steps}
    #-------------------------------------

    # Initialize the optimizer parameters
    optimizer_args = {
        'lr': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        # 'betas': (0.9, 0.98),
        'warmup_init': config['warmup_init'],
        'relative_step': config['relative_step']
    }

    # ----------------------------
    # initialize the bucket samplers for distributed environment
    # boundaries = config['boundaries']
    # batch_sizes = config['batch_sizes']

    # train_sampler = SequenceLengthBatchSampler(train_dataset,
    #                                             boundaries = boundaries,
    #                                             batch_sizes = batch_sizes)

    # test_sampler = SequenceLengthBatchSampler(test_dataset,
    #                                             boundaries = boundaries,
    #                                             batch_sizes = batch_sizes)

    # ------------------------------
    # initialize a bucket sampler with fixed batch size in the case of single machine
    # with parallelization on multiple gpus
    train_sampler = BucketSampler(train_dataset, config['batch_size'])

    test_sampler = BucketSampler(test_dataset, config['batch_size'])
    
    # ------------------------------

    # Initialize the loaders parameters
    train_loader_args = {'batch_sampler': train_sampler, 'collate_fn': partial(collate_fn_trunc, max_len = train_dataset.max_len,
                                                  eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)}

    test_loader_args = {'batch_sampler': test_sampler, 'collate_fn': partial(collate_fn_trunc, max_len = train_dataset.max_len,
                                                  eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)}

    # Add the datasets and hyperparameters to trainer
    trainer.compile(train_dataset, test_dataset, tokenizer, train_loader_args,
                    test_loader_args, optimizer_kwargs = optimizer_args,
                    # lr_scheduler=get_linear_schedule_with_warmup,
                    # lr_scheduler_kwargs=scheduler_args,
                    predict_with_generate = True,
                    stopping_patience = config["stopping_patience"],
                    hugging_face = True,
                    is_distributed=is_distributed,
                    logging_dir=config['logging_dir'],
                    dist=dist
                    )

    # load the model
    trainer.load(config['model_dir'], load_best = not config['continue'])
    
    # Train the model
    trainer.train(config['epochs'] - trainer.current_epoch, auto_save = True, log_step = config['log_step'], saving_directory=config['new_model_dir'], save_best = config['save_best'],
                  metric_for_best_model = config['metric_for_best_model'], metric_objective = config['metric_objective'])
    
    if config['return_trainer']:
        
        return trainer
    
    return None
