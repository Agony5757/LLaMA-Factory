import os
from pathlib import Path

def calculate_ppl(
    model_path, 
    adapter_path, 
    dataset, 
    save_name,
    use_quanta,
    use_qpeft,
    qpeft_arch,
    qpeft_n_qlayers,
    lora_rank
):
    command  = (f"python cal_ppl_lora.py --model_name_or_path {model_path}"
                f" --adapter_name_or_path {adapter_path}"
                f" --dataset {dataset}"
                f" --save_name {save_name}"
                f" --use_quanta {use_quanta}"
                f" --use_qpeft {use_qpeft}"
                f" --qpeft_arch {qpeft_arch}"
                f" --qpeft_n_qlayers {qpeft_n_qlayers}"
                f" --lora_rank {lora_rank}")
    
    print("Run command: ", command)

    ret = os.system(command)
    if ret != 0:
        print('Execution failed, ret = ', ret)
        return False
    
    return True

if __name__ == '__main__':
    basepath_str = "/mnt/share/FinalData/qpeft"
    basepath = Path(basepath_str)

    data_path_list = list()

    for dirpath, dirnames, filenames in os.walk(basepath):
        #print(dirpath)
        #print(dirnames)
        #print(filenames)
        # if 'ABC' not in dirpath and 'BC' not in dirpath:
        #     continue
        if 'quanta' not in dirpath:
            continue
        for dirname in dirnames:
            if dirname.endswith('v3'):
                data_path_list.append(str(Path(dirpath) / dirname))
    
    #print(data_path_list)
    #exit(0)
    model_path = '/mnt/share/Qwen3-1.7B/'
    dataset = 'CPsyCounD_test_100_0'
    savename = 'ppl.json'


    for path in data_path_list:
        adapter_path : str = path[1+len(basepath_str):]
        savepath = f'eval/ppl/{adapter_path}'
        if os.path.exists(savepath):
            print(f"{savepath} exists, just skip.")
            continue
        
        attributes = adapter_path.split('/')

        # qpeft_xxx / quanta / lora
        use_quanta = False
        use_qpeft = False
        qpeft_arch = 'ABC'
        qpeft_n_qlayers = None
        lora_rank = None

        peft_type = attributes[0]
        if 'qpeft' in peft_type:
            use_qpeft = True

            # extract qpeft_args
            # 
            # 1. qpeft_arch: ABC/BC/A/...
            # 2. qpeft_n_qlayers: default or any integer          
            qpeft_args = peft_type.split('_')
            qpeft_arch = qpeft_args[1]
            if qpeft_args[2] != 'default':
                qpeft_n_qlayers = int(qpeft_args[2])
        elif 'quanta' in peft_type:
            use_quanta = True
        
        lora_rank_target_str = attributes[-1]
        lora_rank = int(lora_rank_target_str.split('_')[1][1:])


        os.makedirs(savepath, exist_ok=False)
        
        if not calculate_ppl(
            model_path=model_path,
            adapter_path=path,
            dataset=dataset,
            save_name=f'{savepath}/{savename}',
            use_quanta = use_quanta,
            use_qpeft = use_qpeft,
            qpeft_arch = qpeft_arch,
            qpeft_n_qlayers = qpeft_n_qlayers,
            lora_rank=lora_rank
            ):
            break

            
