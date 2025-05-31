import yaml
import tempfile
import subprocess
import os
import itertools # For product
from typing import Dict, Any, List, Optional, Union
import time # For adding a small delay if needed

# --- Configuration Generation (from previous response, slightly adapted) ---

def generate_output_dir(
    base_output_path: str,
    model_name_or_path: str,
    dataset_name: str, # Expects a single dataset name for path generation
    max_samples: int,
    finetuning_type: str,
    finetuning_type_special: str,
    qpeft_configs: str = "",
    lora_rank: Union[int, None] = None,
    lora_target: Union[str, List[str]] = "all",
    custom_suffix: str = ""
) -> str:
    """
    Generates a structured output directory path.
    Example: {base_output_path}/{lora_type}/{model_short_name}/{dataset_name}/samples_{max_samples}/{finetuning_type}_{target}_r{lora_rank}
    """
    model_short_name = os.path.basename(model_name_or_path).replace("/", "_") # Handle potential slashes
    
    if finetuning_type_special == 'none':
        lora_type = "lora"
    elif finetuning_type_special == 'quanta':
        lora_type = "quanta"
    elif finetuning_type_special == 'qpeft':
        lora_type = "qpeft_" + qpeft_configs
        
    else:
        raise ValueError(f"Invalid finetuning_type_special: {finetuning_type_special}")

    path_parts = [
        base_output_path,
        lora_type,
        model_short_name,
        dataset_name, # Assumes single dataset name for directory structure
        f"samples_{max_samples}",
    ]
    
    type_str = finetuning_type
    if finetuning_type == "lora" and lora_rank is not None:
        type_str += f"_r{lora_rank}"
    
    if lora_target is not None:
        type_str += f"_{lora_target}" # Scientific notation e.g., 5e-4
        
    path_parts.append(type_str + custom_suffix)
    
    return os.path.join(*path_parts)

def generate_training_config(
    # Core model and data settings
    model_name_or_path: str,
    dataset: Union[str, List[str]], # Can be a single dataset name or a comma-separated list
    template: str = "qwen",
    max_samples: int = 1000,
    cutoff_len: int = 1024,
    
    # Finetuning method
    stage: str = "sft",
    finetuning_type: str = "lora", # e.g., "lora", "full"
    lora_rank: Union[int, None] = 4, # Made optional, set based on finetuning_type
    lora_target: Union[str, List[str]] = "all", 
    finetuning_type_special: str = 'none',
    qpeft_arch : str = 'ABC',
    qpeft_n_qlayers : Optional[int] = None,
    
    # Output
    base_output_path: str = "/mnt/share/qpeft", 
    output_dir_suffix: str = "", 
    logging_steps: int = 1, # Increased for less verbose logs during scans
    save_steps: int = 100, # Potentially save less frequently during scans
    plot_loss: bool = True,
    overwrite_output_dir: bool = True,
    report_to: str = "wandb", # Consider "none" if wandb gets too cluttered
    
    # Training hyperparams
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5.0e-4,
    num_train_epochs: float = 6.0,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    bf16: bool = True,
    
    # Evaluation
    val_size: float = 0.1,
    per_device_eval_batch_size: int = 1,
    eval_strategy: str = "steps",
    eval_steps: int = 30, # Align with save_steps or logging_steps
    
    # Other LLaMA Factory specific
    trust_remote_code: bool = True,
    overwrite_cache: bool = True,
    preprocessing_num_workers: int = 16,
    ddp_timeout: int = 180000000,
    resume_from_checkpoint: Union[str, None] = None,
    
    **kwargs: Any 
) -> Dict[str, Any]:
    """
    Generates the configuration dictionary for LLaMA Factory training.
    """
    # Determine the primary dataset name for path generation
    # If 'dataset' is a list, take the first; if a string, split by comma and take first.
    if isinstance(dataset, list):
        primary_dataset_name = dataset[0] if dataset else "unknown_dataset"
    elif isinstance(dataset, str):
        primary_dataset_name = dataset.split(',')[0]
    else:
        primary_dataset_name = "unknown_dataset"

    if finetuning_type_special == 'qpeft':
        if qpeft_n_qlayers is None:
            qpeft_configs = f"{qpeft_arch}_default"
        else:
            qpeft_configs = f"{qpeft_arch}_{qpeft_n_qlayers}"
    else:
        qpeft_configs = ""

    output_dir = generate_output_dir(
        base_output_path=base_output_path,
        model_name_or_path=model_name_or_path,
        dataset_name=primary_dataset_name,
        max_samples=max_samples,
        finetuning_type=finetuning_type,
        finetuning_type_special=finetuning_type_special,
        qpeft_configs=qpeft_configs,
        lora_rank=lora_rank if finetuning_type == "lora" else None,
        lora_target=lora_target, # Pass LR for output path
        custom_suffix=output_dir_suffix
    )

    config = {
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": trust_remote_code,
        
        "stage": stage,
        "do_train": True, 
        "finetuning_type": finetuning_type,
        "save_safetensors": False,
        "use_quanta": True if finetuning_type_special == 'quanta' else False,
        "use_qpeft": True if finetuning_type_special == 'qpeft' else False,
        
        "dataset": dataset if isinstance(dataset, str) else ",".join(dataset), # LLaMA Factory expects comma-separated string
        "template": template,
        "cutoff_len": cutoff_len,
        "max_samples": max_samples,
        "overwrite_cache": overwrite_cache,
        "preprocessing_num_workers": preprocessing_num_workers,
        
        "output_dir": output_dir,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "plot_loss": plot_loss,
        "overwrite_output_dir": overwrite_output_dir,
        "report_to": report_to,
        
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "bf16": bf16,
        "ddp_timeout": ddp_timeout,
        
        "val_size": val_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
    }

    if finetuning_type == "lora":
        if lora_rank is None: # Add a default if not provided for LoRA
            config["lora_rank"] = 4 
        else:
            config["lora_rank"] = lora_rank
        
        if isinstance(lora_target, list): 
            config["lora_target"] = ",".join(lora_target)
        else:
            config["lora_target"] = lora_target
    elif "lora_rank" in config: # Clean up if not lora
        del config["lora_rank"]

    if "lora_target" in config and finetuning_type != "lora":
        del config["lora_target"]

    if finetuning_type_special == 'qpeft':
        config['qpeft_arch'] = qpeft_arch
        if qpeft_n_qlayers is not None:
            config['qpeft_n_qlayers'] = qpeft_n_qlayers
            
    if resume_from_checkpoint:
        config["resume_from_checkpoint"] = resume_from_checkpoint

    config.update(kwargs) # Apply any overrides from scan or specific kwargs
    
    return config

# --- Training Execution (same as before) ---

def run_llama_factory_training(params: dict, cli_command: str = "llamafactory-cli"):
    """
    Runs LLaMA Factory training with the given parameters.
    """
    temp_yaml_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False, encoding='utf-8') as tmp_file:
            yaml.dump(params, tmp_file, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
            temp_yaml_path = tmp_file.name
        
        print(f"临时配置文件已创建: {temp_yaml_path}")
        # print("--- TEMP YAML CONTENT ---")
        # with open(temp_yaml_path, 'r', encoding='utf-8') as f:
        #     print(f.read())
        # print("-------------------------")

        command_list = [cli_command, "train", temp_yaml_path]
        print(f"执行命令: {' '.join(command_list)}")

        process = subprocess.run(command_list, check=True, text=True, encoding='utf-8')
        
        print("训练命令成功执行。")
        return True # Indicate success

    except subprocess.CalledProcessError as e:
        print(f"训练过程中发生错误: {e}")
        if hasattr(e, 'stdout') and e.stdout: print("STDOUT:", e.stdout)
        if hasattr(e, 'stderr') and e.stderr: print("STDERR:", e.stderr)
        return False # Indicate failure
    except Exception as e:
        print(f"发生意外错误: {e}")
        return False # Indicate failure
    finally:
        if temp_yaml_path and os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
            print(f"临时配置文件已删除: {temp_yaml_path}")

# --- Parameter Scanning Function ---

def scan_and_train(
    param_grid: Dict[str, List[Any]],
    base_config_args: Dict[str, Any],
    cli_command: str = "llamafactory-cli",
    dry_run: bool = False # If True, only print configs, don't run
):
    """
    Scans through combinations of parameters, generates configs, and runs training.

    Args:
        param_grid (Dict[str, List[Any]]):
            Dictionary where keys are parameter names (must match args of
            generate_training_config or be in its **kwargs) and values are lists
            of settings to try.
            Example: {"learning_rate": [1e-4, 5e-5], "max_samples": [1000, 2000]}
        base_config_args (Dict[str, Any]):
            Dictionary of base configuration arguments that are fixed across all runs.
            These are passed to generate_training_config.
        cli_command (str): The LLaMA Factory CLI command.
        dry_run (bool): If True, prints configurations but does not execute training.
    """
    keys = list(param_grid.keys())
    value_combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    total_runs = len(value_combinations)
    
    print(f"--- Starting Parameter Scan ---")
    print(f"Base config parameters: {base_config_args}")
    print(f"Scanning over parameters: {param_grid}")
    print(f"Total number of runs: {total_runs}")
    
    successful_runs = 0
    failed_runs = 0

    for i, combo_values in enumerate(value_combinations):
        current_run_params_override = dict(zip(keys, combo_values))
        
        print(f"\n--- Run {i+1}/{total_runs} ---")
        print(f"Current scanned parameters: {current_run_params_override}")

        # Create a full set of arguments for generate_training_config
        # Start with base, then update with current scan parameters
        current_config_gen_args = base_config_args.copy()
        current_config_gen_args.update(current_run_params_override)
        
        try:
            # Generate the specific configuration for this run
            training_params_dict = generate_training_config(**current_config_gen_args)
            
            print("Generated training configuration:")
            # print(yaml.dump(training_params_dict, sort_keys=False, indent=2)) # Pretty print
            print(f"  Output directory: {training_params_dict['output_dir']}")
            print(f"  Learning rate: {training_params_dict.get('learning_rate')}")
            print(f"  Max samples: {training_params_dict.get('max_samples')}")
            print(f"  LoRA rank: {training_params_dict.get('lora_rank', 'N/A')}")

            # Check if output_dir already exists and overwrite_output_dir is False
            if not training_params_dict.get("overwrite_output_dir", True) and \
               os.path.exists(training_params_dict["output_dir"]) and \
               any(os.scandir(training_params_dict["output_dir"])): # Check if not empty
                print(f"Skipping run: Output directory {training_params_dict['output_dir']} exists and is not empty, and overwrite_output_dir is False.")
                # failed_runs +=1 # Or a new category like 'skipped_runs'
                continue

            if dry_run:
                print("Dry run: Skipping actual training execution.")
                successful_runs +=1 # Count as success for dry run
                continue
            
            if run_llama_factory_training(training_params_dict, cli_command):
                print(f"Run {i+1}/{total_runs} COMPLETED successfully.")
                successful_runs += 1
            else:
                print(f"Run {i+1}/{total_runs} FAILED.")
                failed_runs += 1
        
        except Exception as e:
            print(f"Run {i+1}/{total_runs} FAILED with an error during config generation or pre-check: {e}")
            import traceback
            traceback.print_exc()
            failed_runs += 1
            time.sleep(10) # Optional: Add a small delay between runs if needed
        
        print("-" * 70)
        # time.sleep(5) # Optional: Add a small delay between runs if needed

    print(f"\n--- Parameter Scan Finished ---")
    print(f"Total runs attempted: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")

# --- Main Execution Example ---

if __name__ == "__main__":
    # 1. Define the parameters you want to scan over
    parameter_scan_grid = {
        "model_name_or_path": [
            "/mnt/share/Qwen3-1.7B",
            # "/mnt/share/AnotherModel-7B" # Example if you have another model
        ],
        "max_samples": [3000, 300, 600, 1000, ],
        "lora_rank": [4, 6, 8, 10],
        "learning_rate": [5.0e-4],
        "finetuning_type_special": [#'none',
                                    # 'quanta',
                                    'qpeft'
                                    ],
        "dataset": ["CPsyCounD_eval", "medical_o1_sft_Chinese", "r1"],
        #"dataset": ["identity"],
        #"lora_target": ["q_proj,k_proj,v_proj", "q_proj,v_proj", "all"]
        #"lora_target": ["q_proj,v_proj", "q_proj,k_proj,v_proj", ],
        "lora_target": ["q_proj,v_proj" ],
        "qpeft_arch": ['B', 'A', 'AB', 'ABC', 'BC'],
        #"qpeft_n_qlayers": [1,2,3,4],
        # "dataset": ["CPsyCounD", "another_dataset_name"] # If you want to scan datasets
    }

    # 2. Define the base configuration arguments that are fixed for this scan
    # These are arguments to `generate_training_config`
    base_arguments = {
        "dataset": "CPsyCounD_eval", # Fixed dataset for this scan example
        "template": "qwen",
        "finetuning_type": "lora",
        "finetuning_type_special": 'none',
        "lora_target": "q_proj,k_proj,v_proj",
        "num_train_epochs": 3.0, # Shorter epochs for scan illustration
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "bf16": True,
        "overwrite_output_dir": False, # Be careful with this for real scans
        "base_output_path": "/mnt/share/qpeft", # Dedicated base for scans
        # "output_dir_suffix": "_initial_scan" # Add a common suffix for this set of scans
        "logging_steps": 1,
        "save_steps": 100, # Might not want to save checkpoints for every scan run or save less often
        "eval_steps": 30,
    }
    
    # If you want to scan finetuning_type:
    # parameter_scan_grid_ft = {
    #     "finetuning_type": ["lora", "full"],
    #     "learning_rate": [5e-5, 1e-5] # LR might need to be different for full FT
    # }
    # base_arguments_ft = base_arguments.copy()
    # base_arguments_ft["lora_rank"] = None # Explicitly set to None if full FT is an option

    # 3. Run the scan
    # Set dry_run=True to test your configurations without actually training
    scan_and_train(parameter_scan_grid, base_arguments, dry_run=False)
    
    # To run a scan with different finetuning types:
    # scan_and_train(parameter_scan_grid_ft, base_arguments_ft, dry_run=True)

    print("\nScript finished.")

