import subprocess
import time
import os

# ================== 配置区 ==================
# 1. 你的训练命令：建议使用绝对路径或确保环境已激活
# 如果使用 conda 环境，可以写成: "conda run -n your_env_name python train.py"
TRAIN_COMMAND = "python train.py" 

# 2. 监控的 GPU ID：你的 RTX 4090 通常是 0
GPU_ID = 0

# 3. 空闲判断阈值 (MiB)：RTX 4090 (24GB) 建议设为 20000 以上确保足够空间
FREE_THRESHOLD = 14500 

# 4. 轮询间隔 (秒)
CHECK_INTERVAL = 60 

# 5. # 或者最推荐的设置：设为 None，表示只要 GPU 不空闲，就一直等下去
MAX_WAIT_TIME = None
# ============================================

def get_free_memory(gpu_id=0):
    try:
        # 查询 nvidia-smi 获取剩余显存
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader", "-i", str(gpu_id)]
        )
        return int(result.decode().strip())
    except Exception as e:
        print(f"获取显存失败: {e}")
        return 0

print(f"🚀 GPU 监控启动。目标：显存 > {FREE_THRESHOLD} MiB (当前 GPU: {GPU_ID})")
start_time = time.time()

while True:
    free_mem = get_free_memory(GPU_ID)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] 空闲显存: {free_mem} MiB")

    if free_mem >= FREE_THRESHOLD:
        print(f"✅ 显存充足！启动训练命令: {TRAIN_COMMAND}")
        # 使用 os.system 或 subprocess 运行
        os.system(TRAIN_COMMAND)
        break
    
    if MAX_WAIT_TIME and (time.time() - start_time) > MAX_WAIT_TIME:
        print("⏰ 等待超时，自动退出。")
        break
        
    time.sleep(CHECK_INTERVAL)