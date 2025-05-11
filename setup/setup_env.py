# =====================
# File: setup/setup_env.py
# =====================
import os, random, numpy as np, torch

def setup_environment(seed=42):
    os.environ.update({
        "PYTHONHASHSEED": str(seed),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "DGLBACKEND": "pytorch"
    })
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
