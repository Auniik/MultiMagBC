import os


def is_runpod():
    return (
        os.path.exists('/workspace') or 
        'runpod' in os.environ.get('HOSTNAME', '').lower() or
        os.environ.get('RUNPOD_POD_ID') is not None
    )

def is_kaggle():
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

def is_local():
    return not is_runpod() and not is_kaggle()

def get_base_path():
    if is_runpod():
        return "/workspace"
    elif is_kaggle():
        return "/kaggle/input"
    else:
        return "./data"
   