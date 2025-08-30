config = {
    "training_type": "checkpoint",  # Options: "scratch", "resume", "checkpoint"
    "B": 32,  # Batch size, mini-batch-size per GPU
    "block_size": 1024,  # Sequence length
    "global_batch_size": 524288,  # in tokens, from gpt2 paper per batch 5M token processed 589824
    "compile": True,  # Whether to use torch.compile
    "dtype": "bfloat16",  # Options: "float32", 
    "training_data_base_dir" : "/experiment_dir/gpt2_tut/data/train_data",
    "validation_iid_base_dir" : "/experiment_dir/gpt2_tut/data/train_data",
    "validation_ood_base_dir": "/experiment_dir/gpt2_tut/data/validation_data", 
    "checkpoint_dir" : "/experiment_dir/gpt2_tut/checkpoints",
    "checkpoint_path": "/experiment_dir/gpt2_tut/checkpoints/ckpt_multi_gpu_step_468000_valloss_2.9072.pt",
    "training_dataset": "edufineweb-10BT",  # Options: "edufineweb", "edufineweb_ood"
    "learning_rate": 6e-4,  # Learning rate
    'weight_decay': 1e-1,  # Weight decay
    "grad_clip": 1.0,
    "wandb_log": True,
    "max_iterations": 600000,  # Total number of training iterations
    "warmup_iters": 3000,
    "log_interval": 10,  # How many steps to log
    "val_loss_every": 2000,  # How many steps to evaluate
    "val_max_steps": 200,
    "sample_every": 200,  # How many steps to sample
    "run_name": "gpt2-small-multi-4-gpu-final-checkpoint-126000",
    
    
}

single_gpu_config = {
    "training_type": "scratch",  # Options: "scratch", "resume", "checkpoint"
    "B": 32,  # Batch size, mini-batch-size per GPU
    "block_size": 1024,  # Sequence length
    "global_batch_size": 524288,  # in tokens, from gpt2 paper per batch 5M token processed 589824
    "compile": True,  # Whether to use torch.compile
    "dtype": "bfloat16",  # Options: "float32", 
    "training_data_base_dir" : "/experiment_dir/self_exp/gpt2_tut/data/train_data",
    "validation_iid_base_dir" : "/experiment_dir/self_exp/gpt2_tut/data/train_data",
    "validation_ood_base_dir": "/experiment_dir/self_exp/gpt2_tut/data/validation_data", 
    "checkpoint_dir" : "/experiment_dir/self_exp/gpt2_tut/single_checkpoints",
    "training_dataset": "edufineweb-10BT",  # Options: "edufineweb", "edufineweb_ood"
    "learning_rate": 6e-4,  # Learning rate
    'weight_decay': 1e-1,  # Weight decay
    "grad_clip": 1.0,
    "wandb_log": True,
    "max_iterations": 600000,  # Total number of training iterations
    "warmup_iters": 2000,
    "log_interval": 10,  # How many steps to log
    "val_loss_every": 2000,  # How many steps to evaluate
    "val_max_steps": 200,
    "sample_every": 100,  # How many steps to sample
    "run_name": "gpt2-small-single-gpu-full_run"
    
    
}