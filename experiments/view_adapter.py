import torch
import os

# 替换为你的 adapter_model.bin 文件的实际路径
# adapter_model_path = (
#     "/mnt/share/FinalData/qpeft/"
#     "qpeft_ABC_default/Qwen3-1.7B/CPsyCounD_eval/samples_3000/lora_r4_q_proj_v_proj_v3/"
#     "adapter_model.bin")
adapter_model_path = (
    "/mnt/share/FinalData/qpeft/"
    "quanta/Qwen3-1.7B/CPsyCounD_eval/samples_3000/lora_r4_q_proj_v_proj_v3/"
    "pytorch_model.bin"
)

if not os.path.exists(adapter_model_path):
    print(f"Error: File not found at {adapter_model_path}")
else:
    try:
        # 加载 adapter_model.bin 文件
        # map_location='cpu' 是一个好习惯，可以确保无论当前设备是什么，
        # 张量都会被加载到 CPU 上，避免 GPU 显存不足的问题
        adapter_state_dict = torch.load(adapter_model_path, map_location='cpu')

        print(f"Successfully loaded {adapter_model_path}")
        print("\n--- Keys in the adapter_model.bin state_dict ---")
        # 打印所有的键（即 LoRA 层的名称）
        for key in adapter_state_dict.keys():
            print(f"- {key}")

        print("\n--- Details of some specific parameters ---")
        # 遍历 state_dict 并打印每个张量的形状和数据类型
        # 通常 LoRA 层的命名会包含 'lora_A' 或 'lora_B'
        count = 0
        with open("adapter_info.txt", 'w') as fp:
            for key, value in adapter_state_dict.items():
                print(f"Key: {key}", file=fp)
                print(f"  Shape: {value.shape}", file=fp)
                print(f"  Dtype: {value.dtype}", file=fp)
                print(f"  Device: {value.device}", file=fp)
                # 如果你想查看张量的值（通常不建议打印整个大张量）
                # 可以打印一小部分，例如前几行几列
                if value.numel() > 0: # 确保张量不为空
                    if value.ndim >= 2:
                        print(f"  Value (top-left 2x2): \n{value[:2, :2]}", file=fp)
                    elif value.ndim == 1:
                        print(f"  Value (first 5 elements): \n{value[:5]}", file=fp)
                print("-" * 30)
                # count += 1
                # if count >= 5: # 只打印前5个键的详细信息，避免输出过多
                #     # print("... (showing only first 5 entries, remove this limit to see all)")
                #     # break
                #     print(" ... press enter to see more")
                #     input()
                #     continue

    except Exception as e:
        print(f"An error occurred while loading or inspecting the file: {e}")
        print("This might happen if the file is corrupted or not a valid PyTorch state_dict.")

