from datetime import datetime
import os
import subprocess


if __name__ == "__main__":
    # Define log folder and script name
    log_folder = './logs'
    script_name = 'main_3d.py'
    os.makedirs(log_folder, exist_ok=True)

    # Run in background
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"{script_name}_{datetime_str}.log")
    cmd = f"nohup python3 {script_name} > '{log_file}' 2>&1 &"
    print(f"Running in background: {cmd}")
    subprocess.Popen(cmd, shell=True)
