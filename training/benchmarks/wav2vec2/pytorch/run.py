import subprocess
import os
import sys
processes = []
dist_world_size = 2
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
current_env = os.environ.copy()
current_env["MASTER_ADDR"] = "10.1.2.155"
current_env["MASTER_PORT"] = "29501"
current_env["WORLD_SIZE"] = str(dist_world_size)
current_env["NODE_RANK"] = str(0)
for local_rank in range(0, 2):
    dist_rank = local_rank
    current_env["RANK"] = str(dist_rank)
    print("dis",dist_rank)
    current_env["LOCAL_RANK"] = str(dist_rank)
    start_cmd = sys.executable + " training/benchmarks/wav2vec2/pytorch/test.py --extern_config_dir /workspace/wav2vec2/wav2vec2_Perf/training/nvidia/wav2vec2-pytorch/config --extern_config_file config_A100x1x8.py --data_dir /workspace/wav2vec2/wav2vec2_data/LibriSpeech --vendor nvidia " 
    
    process = subprocess.Popen(start_cmd, shell=True, env=current_env)
# process = subprocess.Popen(start_cmd, shell=True)
    processes.append(process)

for proc in processes:
    proc.wait()


# print(int(os.environ.get('WORLD_SIZE',36)))
print(os.environ.get('WORLD_SIZE'))
