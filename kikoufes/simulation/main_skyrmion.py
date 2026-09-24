import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from prcpy.RC.Pipeline_RC import *
from prcpy.TrainingModels.RegressionModels import *
from prcpy.Maths.Target_functions import get_npy_data
from reservoir_metrics import get_complexity
from reservoir_metrics import get_readout_matrix

def run_model(data_dir_path, target_path, process_params, model_params):
    print("[1/7] RC pipeline start", flush=True)
    print(f"[1/7] Loading dataset from: {data_dir_path}", flush=True)

    rc_pipeline = Pipeline(str(data_dir_path), "scan", process_params)
    print("[2/7] Preprocessing dataset...", flush=True)
    rc_df = rc_pipeline.get_rc_df()
    print(f"前処理後のRCデータ形状 = {rc_df.shape} (サンプル数, 特徴次元数)", flush=True)
    print(f"RC演算に使う1サンプルあたりの次元数 = {rc_df.shape[1]}", flush=True)

    print(f"[3/7] Loading target from: {target_path}", flush=True)
    target_values = get_npy_data(str(target_path), norm=True)
    rc_pipeline.define_target(target_values)

    print("[4/7] Defining input sequence and measuring reservoir statistics...", flush=True)
    rc_pipeline.define_input(target_values[:500])
    nl = rc_pipeline.get_non_linearity()
    lmc = rc_pipeline.get_linear_memory_capacity(
        kmax=12, remove_auto_correlation=True
    )[0]
    X = get_readout_matrix(rc_pipeline)   # target列を除いたreadout行列 (n_scans, n_features)
    CP, (er1, er2) = get_complexity(X, split="half")

    print(f"NL = {nl}", flush=True)
    print(f"LMC = {lmc}", flush=True)
    print(f"CP = {CP:.3f}", flush=True)
  

    print("[5/7] Defining ridge regression model...", flush=True)
    model = define_Ridge(model_params)
    rc_params = {
        "model": model,
        "tau": 10,
        "test_size": 0.3,
        "error_type": "MSE",
    }

    print("[6/7] Running RC training and prediction...", flush=True)
    rc_pipeline.run(rc_params)
    results = rc_pipeline.get_rc_results()
    train_mse = results["error"]["train_error"]
    test_mse = results["error"]["test_error"]
    print(f"Training MSE = {format(train_mse, '0.3e')}", flush=True)
    print(f"Testing MSE = {format(test_mse, '0.3e')}", flush=True)

    print("[7/7] Preparing results plot...", flush=True)
    train_ts = np.arange(results["train"]["y_train"].shape[0])
    test_ts = np.arange(results["test"]["y_test"].shape[0])
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(data_dir_path.name, size=20)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(train_ts, results["train"]["y_train"], label="Train")
    ax1.plot(train_ts, results["train"]["train_pred"], label="Train predict")
    ax1.set_title(f"Training MSE: {format(train_mse, '0.3e')}")
    ax1.legend()

    ax2.plot(test_ts, results["test"]["y_test"], label="Test")
    ax2.plot(test_ts, results["test"]["test_pred"], label="Test predict")
    ax2.set_title(f"Testing MSE: {format(test_mse, '0.3e')}")
    ax2.legend()
    fig.tight_layout()

    return {
        "model": data_dir_path.name,
        "training_mse": train_mse,
        "testing_mse": test_mse,
        "nl": nl,
        "lmc": lmc,
        "cp": CP,
    }


if __name__ == "__main__":
    data_root = Path("data_full") / "Cu2OSeO3"
    model_names = [
        "SkSplit",
        "SkUnified",
    ]
    target_path = "../PRCpy/data_full/chaos/mackey_glass_t17.npy"

    process_params = {
        "Xs": "Frequency",
        "Readouts": "Spectra",
        "remove_bg": False,
        "smooth": True,
        "smooth_win": 51,
        "smooth_rank": 4,
        "cut_xs": False,
        "x1": 2,
        "x2": 5,
        "normalize_local": False,
        "normalize_global": True,
        "sample": True,
        "sample_rate": 8,
        "delimiter": ",",
        "transpose": False,
    }

    model_params = {
        "alpha": 1e-1,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 0.0001,
        "solver": "auto",
        "positive": False,
        "random_state": None,
    }

    summary = []
    for model_name in model_names:
        data_dir_path = data_root / model_name
        if not data_dir_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir_path}")
        print(f"\n===== {model_name} =====", flush=True)
        summary.append(run_model(data_dir_path, target_path, process_params, model_params))

    print("\n===== Summary of all models =====", flush=True)
    print(
        f"{'Model':<10} {'Training MSE':>15} {'Testing MSE':>15} "
        f"{'NL':>12} {'LMC':>12} {'CP':>12}",
        flush=True,
    )
    print("-" * 68, flush=True)
    for result in summary:
        print(
            f"{result['model']:<10} "
            f"{result['training_mse']:>15.6e} "
            f"{result['testing_mse']:>15.6e} "
            f"{result['nl']:>12.6f} "
            f"{result['lmc']:>12.6f} "
            f"{result['cp']:>12.6f}",
            flush=True,
        )

    plt.show()