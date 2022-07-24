def prep_env():
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": ".",
        "filename": "wtbdata_245days.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "input_len": 144,
        "output_len": 288,
        "start_col": 3,
        "hidR": 96,
        "in_var": 10,
        "out_var": 1,
        "day_len": 144,
        "train_size":214,
        "val_size": 31,
        "total_size": 245,
        "lstm_layer": 1,
        "dropout": 0.05,
        "num_workers": 16,
        "train_epochs": 10,
        "batch_size": 64,
        "patience": 5,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "use_gpu": True,
        "gpu": 0,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "pytorch",
        "is_debug": True
    }
    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
