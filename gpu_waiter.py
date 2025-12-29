import subprocess
import time
import os
import sys  # 导入 sys 以读取参数

# ================== 配置区 ==================
# 1. 显存判断阈值 (MiB)：RTX 4090 建议设为 14500 以上
FREE_THRESHOLD = 14500 

# 2. 监控的 GPU ID
GPU_ID = 0

# 3. 轮训间隔 (秒)
CHECK_INTERVAL = 60 

# 4. 超时设置
MAX_WAIT_TIME = None

# --- [联动核心：动态获取命令] ---
# 如果执行时带了参数（如 python gpu_waiter.py python train.py --alpha 0.5）
# 则拼接这些参数作为运行命令；否则使用默认值
if len(sys.argv) > 1:
    TRAIN_COMMAND = " ".join(sys.argv[1:])
else:
    TRAIN_COMMAND = "python train.py" 
# ============================================

def get_free_memory(gpu_id=0):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader", "-i", str(gpu_id)]
        )
        return int(result.decode().strip())
    except Exception as e:
        print(f"获取显存失败: {e}")
        return 0

print(f"🚀 GPU 监控启动。目标：显存 > {FREE_THRESHOLD} MiB (当前 GPU: {GPU_ID})")
print(f"📦 待运行任务: {TRAIN_COMMAND}")
start_time = time.time()

while True:
    free_mem = get_free_memory(GPU_ID)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    if free_mem >= FREE_THRESHOLD:
        print(f"\n[{current_time}] ✅ 显存充足 ({free_mem} MiB)！正在启动任务...")
        # 联动点：执行传入的消融实验命令
        os.system(TRAIN_COMMAND)
        break
    else:
        # 使用 \r 实现单行刷新，避免 log 文件刷屏
        print(f"[{current_time}] ⏳ 显存不足 ({free_mem}/{FREE_THRESHOLD} MiB)，继续等待...", end='\r')
        
    if MAX_WAIT_TIME and (time.time() - start_time) > MAX_WAIT_TIME:
        print("\n⏰ 等待超时，自动退出。")
        break
        
    time.sleep(CHECK_INTERVAL)