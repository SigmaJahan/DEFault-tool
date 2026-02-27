from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "d_DEFault"
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "default_models"

DETECTION_DATASET = Path("A_Detection/1st_Level_OverSampled.csv")
STATIC_DATASET = Path("C_RootCauseAnalysis/static_features_df.csv")

DYNAMIC_TRAINING_SPECS = {
    "detection": {
        "relative_path": DETECTION_DATASET,
        "positive_label": "buggy",
    },
    "activation": {
        "relative_path": Path("B_Categorization/Training_Data/activation_correct_data.csv"),
        "positive_label": "Activation",
    },
    "layer": {
        "relative_path": Path("B_Categorization/Training_Data/layer_correct_data.csv"),
        "positive_label": "Layer",
    },
    "hyperparameter": {
        "relative_path": Path("B_Categorization/Training_Data/hyperparameter_correct_data.csv"),
        "positive_label": "Hyperparameter",
    },
    "loss": {
        "relative_path": Path("B_Categorization/Training_Data/loss_correct_data.csv"),
        "positive_label": "Loss",
    },
    "optimization": {
        "relative_path": Path("B_Categorization/Training_Data/Optimization_correct_data.csv"),
        "positive_label": "Optimization",
    },
    "regularizer": {
        "relative_path": Path("B_Categorization/Training_Data/regularizer_correct_data.csv"),
        "positive_label": "Regularizer",
    },
    "weights": {
        "relative_path": Path("B_Categorization/Training_Data/weight_correct_data.csv"),
        "positive_label": "Weight",
    },
}

DYNAMIC_MODEL_ORDER = [
    "detection",
    "activation",
    "layer",
    "hyperparameter",
    "loss",
    "optimization",
    "regularizer",
    "weights",
]

DYNAMIC_FEATURE_COLUMNS = [
    "gpu_memory_utilization",
    "cpu_utilization",
    "train_acc",
    "val_acc",
    "memory_usage",
    "loss_oscillation",
    "acc_gap_too_big",
    "adjusted_lr",
    "dying_relu",
    "gradient_vanish",
    "gradient_explode",
    "gradient_median",
    "std_grad",
    "gradient_min",
    "mean_grad",
    "large_weight_count",
    "mean_activation",
    "std_activation",
    "gradient_std",
    "gradient_max",
]

STATIC_TARGET_COLUMN = "Buggy"
STATIC_ID_COLUMN = "Model_File"

KERAS_LAYER_TYPES = [
    "Dense",
    "Activation",
    "Dropout",
    "Flatten",
    "InputLayer",
    "Conv1D",
    "Conv2D",
    "Conv3D",
    "MaxPooling1D",
    "MaxPooling2D",
    "MaxPooling3D",
    "SimpleRNN",
    "GRU",
    "LSTM",
    "Embedding",
    "BatchNormalization",
    "LayerNormalization",
]

KERAS_ACTIVATIONS = [
    "softmax",
    "relu",
    "tanh",
    "sigmoid",
    "hard_sigmoid",
    "exponential",
    "linear",
]

ROOT_CAUSE_HINTS = {
    "CountDense": "Check Dense layer count and placement.",
    "CountActivation": "Check activation placement by layer.",
    "CountDropout": "Review dropout usage and rates.",
    "CountConv1D": "Inspect Conv1D configuration and connectivity.",
    "CountConv2D": "Inspect Conv2D configuration and connectivity.",
    "CountConv3D": "Inspect Conv3D configuration and connectivity.",
    "CountMaxPooling1D": "Verify MaxPooling1D settings.",
    "CountMaxPooling2D": "Verify MaxPooling2D settings.",
    "CountMaxPooling3D": "Verify MaxPooling3D settings.",
    "CountSimpleRNN": "Check SimpleRNN layer arguments and sequencing.",
    "CountGRU": "Check GRU layer arguments and sequencing.",
    "CountLSTM": "Check LSTM layer arguments and sequencing.",
    "Countsoftmax": "Softmax usage may be incorrect for output formulation.",
    "Countrelu": "Review ReLU placement and dead-neuron risk.",
    "Counttanh": "Review tanh placement and saturation risk.",
    "Countsigmoid": "Review sigmoid usage for task type.",
    "CountL1": "Check L1 regularization usage.",
    "CountL2": "Check L2 regularization usage.",
    "CountL1L2": "Check combined L1/L2 regularization usage.",
    "Total_Params": "Parameter count may be inconsistent with task complexity.",
    "Trainable_Params": "Trainable parameter count may be incorrect.",
    "Num_Neurons": "Neuron counts across layers may be mismatched.",
    "Max_Neurons": "A layer may be over/under-sized.",
    "Max_Dimension_Output": "Output dimensionality may be incompatible downstream.",
    "Input_Output_Mismatch_Count_Layer": "Layer shape transitions appear inconsistent.",
}

