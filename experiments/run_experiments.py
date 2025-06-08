import yaml
import tempfile
import subprocess
import os
import itertools
from typing import Dict, Any, List, Optional, Union

def generate_config_list(    
    model_name_or_path: str,    
    dataset_name: str,
    max_samples: int,
    finetuning_type: str,
    finetuning_type_special: str,
    qpeft_configs: str = "",
    lora_rank: Union[int, None] = None,
    lora_target: Union[str, List[str]] = "all",
    custom_suffix: str = ""
) -> list[str]:
    
    model_short_name = os.path.basename(model_name_or_path).replace("/", "_")
    if finetuning_type_special == 'none':
        lora_type = "lora"
    elif finetuning_type_special == 'quanta':
        lora_type = "quanta"
    elif finetuning_type_special == 'qpeft':
        lora_type = "qpeft_" + qpeft_configs
    else:
        raise ValueError(f"Invalid finetuning_type_special: {finetuning_type_special}")
    path_parts = [
        lora_type,
        model_short_name,
        dataset_name,
        f"samples_{max_samples}",
    ]
    
    type_str = finetuning_type
    if finetuning_type == "lora" and lora_rank is not None:
        type_str += f"_r{lora_rank}"
    
    if lora_target is not None:
        type_str += f"_{lora_target}"
    
    path_parts.append(type_str + custom_suffix)

    return path_parts

# --- 配置生成 ---
def generate_output_dir(
    base_output_path: str,
    model_name_or_path: str,
    dataset_name: str,
    max_samples: int,
    finetuning_type: str,
    finetuning_type_special: str,
    qpeft_configs: str = "",
    lora_rank: Union[int, None] = None,
    lora_target: Union[str, List[str]] = "all",
    custom_suffix: str = ""
) -> str:
    
    path_parts = [base_output_path]
    path_parts.extend(generate_config_list(
        model_name_or_path=model_name_or_path,
        dataset_name=dataset_name,
        max_samples=max_samples,
        finetuning_type=finetuning_type,
        finetuning_type_special=finetuning_type_special,
        qpeft_configs=qpeft_configs,
        lora_rank=lora_rank,
        lora_target=lora_target,
        custom_suffix=custom_suffix
    ))
    
    return os.path.join(*path_parts)

def generate_training_config(
    model_name_or_path: str,
    dataset: Union[str, List[str]],
    template: str = "qwen",
    max_samples: int = 1000,
    cutoff_len: int = 1024,
    stage: str = "sft",
    finetuning_type: str = "lora",
    lora_rank: Union[int, None] = 4,
    lora_target: Union[str, List[str]] = "all",
    finetuning_type_special: str = 'none',
    qpeft_arch : str = 'ABC',
    qpeft_n_qlayers : Optional[int] = None,
    base_output_path: str = "/mnt/share/qpeft", 
    output_dir_suffix: str = "", 
    logging_steps: int = 1,
    save_steps: int = 100,
    plot_loss: bool = True,
    overwrite_output_dir: bool = True,
    report_to: str = "wandb",
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5.0e-4,
    num_train_epochs: float = 6.0,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    bf16: bool = True,
    val_size: float = 0.1,
    per_device_eval_batch_size: int = 1,
    eval_strategy: str = "steps",
    eval_steps: int = 30,
    trust_remote_code: bool = True,
    overwrite_cache: bool = True,
    preprocessing_num_workers: int = 16,
    ddp_timeout: int = 180000000,
    resume_from_checkpoint: Union[str, None] = None,
    **kwargs: Any 
) -> Dict[str, Any]:
    if isinstance(dataset, list):
        primary_dataset_name = dataset[0] if dataset else "unknown_dataset"
    elif isinstance(dataset, str):
        primary_dataset_name = dataset.split(',')[0]
    else:
        primary_dataset_name = "unknown_dataset"

    if finetuning_type_special == 'qpeft':
        qpeft_configs = f"{qpeft_arch}_{qpeft_n_qlayers}" if qpeft_n_qlayers else f"{qpeft_arch}_default"
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
        lora_target=lora_target,
        custom_suffix=output_dir_suffix
    )

    config = {
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": trust_remote_code,
        "stage": stage,
        "do_train": True, 
        "finetuning_type": finetuning_type,
        "save_safetensors": False,
        "use_quanta": finetuning_type_special == 'quanta',
        "use_qpeft": finetuning_type_special == 'qpeft',
        "dataset": dataset if isinstance(dataset, str) else ",".join(dataset),
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
        config["lora_rank"] = lora_rank if lora_rank is not None else 4
        if isinstance(lora_target, list): 
            config["lora_target"] = ",".join(lora_target)
        else:
            config["lora_target"] = lora_target
    elif "lora_rank" in config:
        del config["lora_rank"]

    if "lora_target" in config and finetuning_type != "lora":
        del config["lora_target"]

    if finetuning_type_special == 'qpeft':
        config['qpeft_arch'] = qpeft_arch
        if qpeft_n_qlayers is not None:
            config['qpeft_n_qlayers'] = qpeft_n_qlayers
            
    if resume_from_checkpoint:
        config["resume_from_checkpoint"] = resume_from_checkpoint

    config.update(kwargs)
    return config

# --- 功能1: 执行训练 ---
def run_training_from_config(params: dict):
    """执行llamafactory-cli train命令使用YAML配置"""
    temp_yaml_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False, encoding='utf-8') as tmp_file:
            yaml.dump(params, tmp_file, sort_keys=False, allow_unicode=True)
            temp_yaml_path = tmp_file.name
        
        command = ["llamafactory-cli", "train", temp_yaml_path]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False
    finally:
        if temp_yaml_path and os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

# --- 功能2: 生成配置文件 ---
def save_config_to_file(
    model_name_or_path: str,
    dataset: Union[str, List[str]],
    template: str = "qwen",
    max_samples: int = 1000,
    cutoff_len: int = 1024,
    stage: str = "sft",
    finetuning_type: str = "lora",
    lora_rank: Union[int, None] = 4,
    lora_target: Union[str, List[str]] = "all",
    finetuning_type_special: str = 'none',
    qpeft_arch : str = 'ABC',
    qpeft_n_qlayers : Optional[int] = None,
    base_output_path: str = "/mnt/share/qpeft", 
    output_dir_suffix: str = "", 
    logging_steps: int = 1,
    save_steps: int = 100,
    plot_loss: bool = True,
    overwrite_output_dir: bool = True,
    report_to: str = "wandb",
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5.0e-4,
    num_train_epochs: float = 6.0,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    bf16: bool = True,
    val_size: float = 0.1,
    per_device_eval_batch_size: int = 1,
    eval_strategy: str = "steps",
    eval_steps: int = 30,
    trust_remote_code: bool = True,
    overwrite_cache: bool = True,
    preprocessing_num_workers: int = 16,
    ddp_timeout: int = 180000000,
    resume_from_checkpoint: Union[str, None] = None,
    output_dir: str = "./configs",
    params: Dict[str, Any] = {},
    **kwargs: Any 
):
    """保存配置到文件并使用配置内容命名"""
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(dataset, list):
        primary_dataset_name = dataset[0] if dataset else "unknown_dataset"
    elif isinstance(dataset, str):
        primary_dataset_name = dataset.split(',')[0]
    else:
        primary_dataset_name = "unknown_dataset"

    if finetuning_type_special == 'qpeft':
        qpeft_configs = f"{qpeft_arch}_{qpeft_n_qlayers}" if qpeft_n_qlayers else f"{qpeft_arch}_default"
    else:
        qpeft_configs = ""
    
    config_list = generate_config_list(
        model_name_or_path=model_name_or_path,
        dataset_name=primary_dataset_name,
        max_samples=max_samples,
        finetuning_type=finetuning_type,
        finetuning_type_special=finetuning_type_special,
        qpeft_configs=qpeft_configs,
        lora_rank=lora_rank,
        lora_target=lora_target,
        custom_suffix=output_dir_suffix
    )
    # 构造基础文件名
    filename = "_".join(config_list)
        
    # 添加后缀并保存
    filename += ".yaml"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(params, f, sort_keys=False, allow_unicode=True)
    
    return filepath

from enum import Enum

class ScanMode(Enum):
    DRY_RUN = "dry_run"
    TRAIN = "train"
    SAVE_CONFIG = "save_config"

# --- 参数扫描 ---
def scan(
    kv_combinations,
    base_config_args: Dict[str, Any],
    scan_mode: ScanMode = ScanMode.TRAIN,
):    
    print(f"开始参数扫描: {len(kv_combinations)} 个配置")
    total_runs = len(kv_combinations)         
    
    for i, combo_kv in enumerate(kv_combinations):
        current_params = base_config_args.copy()
        current_params.update(combo_kv)
        
        training_config = generate_training_config(**current_params)
        
        print(f"\n {i+1}/{total_runs}: {training_config['output_dir']}")
        
        if scan_mode == ScanMode.SAVE_CONFIG:
            save_config_to_file(**current_params, params=training_config, output_dir="./configs")
        elif scan_mode == ScanMode.DRY_RUN:
            print("DRY RUN: 跳过训练")
            continue
        elif scan_mode == ScanMode.TRAIN:
            if not run_training_from_config(training_config):
                print(f"运行失败: {training_config['output_dir']}")
        else:
            raise ValueError(f"Invalid scan_mode: {scan_mode}")

def scan_combinations(
    param_grid: Dict[str, List[Any]],
):
    keys = list(param_grid.keys())
    kv_combinations = []
    for combo_values in itertools.product(*(param_grid[k] for k in keys)):
        kv_combinations.append(dict(zip(keys, combo_values)))

    return kv_combinations
        

# --- 使用示例 ---
if __name__ == "__main__":

    base_arguments = {
        "model_name_or_path": 
            #"/data/home/agony/models/Qwen3-4B",
            #"/data/home/agony/models/Qwen3-8B",
            "/mnt/share/Qwen3-1.7B",        
        "learning_rate": 5.0e-5,
        "dataset": "CPsyCounD_eval", # Fixed dataset for this scan example
        "template": "qwen",
        "finetuning_type": "lora",
        "finetuning_type_special": 'none',
        "lora_target": "q_proj,k_proj,v_proj",
        "num_train_epochs": 3.0, # Shorter epochs for scan illustration
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "bf16": True,
        "overwrite_output_dir": False, # Be careful with this for real scans
        "base_output_path": "./qpeft", # Dedicated base for scans
        "output_dir_suffix": "_v3", # Add a common suffix for this set of scans
        "logging_steps": 1,
        "save_steps": 100, # Might not want to save checkpoints for every scan run or save less often
        "eval_steps": 30,
    }

    parameter_scan_grid1 = {
        "max_samples": [
            3000, 
            #300, 
            #600, 
            #1000, 
        ],
        "lora_rank": [
            4, 
            6, 
            8, 
            #10
        ],
        "finetuning_type_special": [
            #'none',
            'quanta',
            #'qpeft'
        ],
        "dataset": [
            "CPsyCounD_eval", 
            #"medical_o1_sft_Chinese", 
            #"r1",
            # "identity",
        ],
        #"lora_target": ["q_proj,k_proj,v_proj", "q_proj,v_proj", "all"]
        #"lora_target": ["q_proj,v_proj", "q_proj,k_proj,v_proj", ],
        "lora_target": [
            "q_proj,v_proj",
            # "q_proj,k_proj,v_proj"
            # "all"
        ],
        # "qpeft_arch": ['B', 'A', 'AB', 'ABC', 'BC'],
        # "qpeft_n_qlayers": [1,2,3,4],
    }

    parameter_scan_grid2 = {
        "max_samples": [
            3000, 
            #300, 
            #600, 
            #1000, 
        ],
        "lora_rank": [
            4, 
            6, 
            8, 
            #10
        ],
        "learning_rate": [5.0e-5],
        "finetuning_type_special": [
            # 'none',
            # 'quanta',
            'qpeft'
        ],
        "dataset": [
            "CPsyCounD_eval", 
            #"medical_o1_sft_Chinese", 
            #"r1",
            # "identity",
        ],
        #"lora_target": ["q_proj,k_proj,v_proj", "q_proj,v_proj", "all"]
        #"lora_target": ["q_proj,v_proj", "q_proj,k_proj,v_proj", ],
        "lora_target": [
            "q_proj,v_proj",
            # "q_proj,k_proj,v_proj"
            # "all"
        ],
        "qpeft_arch": [
            # 'B', 
            #'ABC', 
            #'AB', 
            'BC',
            'A', 
        ],
        # "qpeft_n_qlayers": [1,2,3,4],
    }
    
    #scan_mode = ScanMode.SAVE_CONFIG
    scan_mode = ScanMode.TRAIN
    #scan_mode = ScanMode.DRY_RUN

    combines1 = scan_combinations(parameter_scan_grid1)
    combines2 = scan_combinations(parameter_scan_grid2)

    #combines = combines1 + combines2
    combines = combines1

    print(combines)

    scan(combines, base_arguments, scan_mode=scan_mode)
