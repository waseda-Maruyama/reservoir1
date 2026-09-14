import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from prcpy.RC.Pipeline_RC import *
from prcpy.TrainingModels.RegressionModels import *
from prcpy.Maths.Target_functions import get_npy_data

if __name__ == "__main__":
    print("[1/7] RC pipeline start", flush=True)

    # Loading data
    data_dir_path = "data_full\Cu2OSeO3\skyrmion"
    prefix = "scan"
    print(f"[1/7] Loading dataset from: {data_dir_path}", flush=True)

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
        "sample_rate": 16,
        "delimiter": ',',
        "transpose": False
    }

    rc_pipeline = Pipeline(data_dir_path,prefix,process_params)
    print("[2/7] Preprocessing dataset...", flush=True)

    rc_df = rc_pipeline.get_rc_df()
    print(f"前処理後のRCデータ形状 = {rc_df.shape} (サンプル数, 特徴次元数)", flush=True)
    print(f"RC演算に使う1サンプルあたりの次元数 = {rc_df.shape[1]}", flush=True)

    # Mackey Glass target generation (prediction)
    mg_path = "../PRCpy/data_full/chaos/mackey_glass_t17.npy"
    print(f"[3/7] Loading target from: {mg_path}", flush=True)
    target_values = get_npy_data(mg_path, norm=True)
    rc_pipeline.define_target(target_values)

    print("[4/7] Defining input sequence and measuring reservoir statistics...", flush=True)
    rc_pipeline.define_input(target_values[:500])
    print(f"NL = {rc_pipeline.get_non_linearity()}", flush=True)
    print(f"LMC = {rc_pipeline.get_linear_memory_capacity(kmax=12, remove_auto_correlation=True)[0]}", flush=True)

    # Define model parameters
    print("[5/7] Defining ridge regression model...", flush=True)
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

    # Define the training model
    model = define_Ridge(model_params)

    # Define the RC parameters
    rc_params = {
        "model": model,
        "tau": 10,
        "test_size": 0.3,
        "error_type": "MSE"
    }

    print("[6/7] Running RC training and prediction...", flush=True)
    rc_pipeline.run(rc_params)

    # Get the results
    print("[6/7] Collecting RC results...", flush=True)
    results = rc_pipeline.get_rc_results()

    # Results
    train_ts = np.arange(results["train"]["y_train"].shape[0])
    test_ts = np.arange(results["test"]["y_test"].shape[0])
    train_ys = results["train"]["y_train"]
    test_ys = results["test"]["y_test"]
    train_preds = results["train"]["train_pred"]
    test_preds = results["test"]["test_pred"]

    # Errors
    train_MSE = results["error"]["train_error"]
    test_MSE = results["error"]["test_error"]

    print(f"Training MSE = {format(train_MSE, '0.3e')}", flush=True)
    print(f"Testing MSE = {format(test_MSE, '0.3e')}", flush=True)
    print("[7/7] Plotting results...", flush=True)

    # Plot results
    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{}".format(os.path.basename(data_dir_path)),size=20)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(train_ts,train_ys,label="Train")
    ax1.plot(train_ts,train_preds,label="Train predict")
    ax1.set_title(f"Training MSE: {format(train_MSE, '0.3e')}")
    ax1.legend()

    ax2.plot(test_ts, test_ys, label="Test")
    ax2.plot(test_ts, test_preds, label="Test predict")
    ax2.set_title(f"Testing MSE: {format(test_MSE, '0.3e')}")
    ax2.legend()

    fig.tight_layout()

    plt.show()

