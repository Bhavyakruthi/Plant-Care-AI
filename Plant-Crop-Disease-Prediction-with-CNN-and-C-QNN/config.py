"""
Configuration for Quantum-Classical Plant Disease Detection
Defines hyperparameters and settings for all four modules
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    dataset_path: str = "./PlantVillage"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Classical preprocessing
    classical_image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Quantum preprocessing (MUST be power of 2 for amplitude encoding)
    quantum_image_size: Tuple[int, int] = (8, 8)  # 64 features -> 6 qubits
    # Alternative: (16, 16) -> 256 features -> 8 qubits
    
    # Augmentation settings
    use_augmentation: bool = True
    augmentation_params: dict = field(default_factory=lambda: {
        'rotation_degrees': 10,
        'horizontal_flip': 0.5,
        'vertical_flip': 0.3,
        'color_jitter': {'brightness': 0.2, 'contrast': 0.2},
    })
    
    batch_size_classical: int = 32
    batch_size_quantum: int = 8  # Smaller for quantum simulators
    num_workers: int = 4
    seed: int = 42


@dataclass
class QCNNConfig:
    """QCNN architecture configuration"""
    n_qubits: int = 6  # For 8x8 images (64 pixels)
    conv_layers: int = 2
    encoding_type: str = "amplitude"  # "amplitude" or "angle"
    
    # Circuit design
    entanglement_pattern: str = "ring"  # "ring", "full", "nearest"
    rotation_gates: List[str] = field(default_factory=lambda: ["RY", "RZ"])
    
    # Pooling strategy
    pooling_type: str = "measurement"  # "measurement" or "trace"
    
    # Output
    n_output_qubits: int = 3  # Features extracted
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # "adam", "rmsprop", "spsa", "cobyla"


@dataclass
class QNNConfig:
    """Quantum Neural Network (VQC) configuration"""
    n_qubits: int = 5
    n_layers: int = 3
    encoding_type: str = "angle"  # "angle" or "amplitude"
    
    # Circuit architecture
    entanglement_pattern: str = "ring"
    rotation_gates: List[str] = field(default_factory=lambda: ["RY", "RZ"])
    
    # Observable
    observable_type: str = "pauliz"  # "pauliz" per qubit
    
    # Training
    learning_rate: float = 0.005
    optimizer: str = "adam"
    

@dataclass
class CNNConfig:
    """Classical CNN configuration"""
    architecture: str = "resnet18"  # "resnet18", "resnet34", "mobilenetv2"
    pretrained: bool = True
    freeze_backbone: bool = False
    freeze_layers: int = 0  # Number of initial layers to freeze
    
    # Feature extraction
    feature_dim: int = 512  # Output dimension before classifier
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    momentum: float = 0.9  # For SGD
    
    # LR scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_params: dict = field(default_factory=lambda: {
        'T_max': 100,  # For cosine
        'step_size': 30,  # For step
        'gamma': 0.1,
    })


@dataclass
class DNNConfig:
    """Dense Neural Network configuration"""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3
    activation: str = "relu"  # "relu", "gelu", "leaky_relu"
    batch_norm: bool = True


@dataclass
class TrainingConfig:
    """General training configuration"""
    num_epochs: int = 100
    early_stopping_patience: int = 15
    
    # Logging
    log_interval: int = 10
    save_dir: str = "./experiments"
    experiment_name: str = "plant_disease_qml"
    
    # Device
    device: str = "cuda"  # "cuda", "cpu"
    mixed_precision: bool = False  # For classical models
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False


@dataclass
class Module1Config:
    """QCNN + DNN configuration"""
    name: str = "QCNN_DNN"
    qcnn: QCNNConfig = field(default_factory=QCNNConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class Module2Config:
    """CNN + QNN configuration"""
    name: str = "CNN_QNN"
    cnn: CNNConfig = field(default_factory=CNNConfig)
    qnn: QNNConfig = field(default_factory=QNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Interface between CNN and QNN
    cnn_to_qnn_dim: int = 16  # Reduce CNN features to this dimension


@dataclass
class Module3Config:
    """QCNN + QNN configuration (Fully Quantum)"""
    name: str = "QCNN_QNN"
    qcnn: QCNNConfig = field(default_factory=QCNNConfig)
    qnn: QNNConfig = field(default_factory=lambda: QNNConfig(n_qubits=3))
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class Module4Config:
    """CNN + DNN configuration (Classical Baseline)"""
    name: str = "CNN_DNN"
    cnn: CNNConfig = field(default_factory=CNNConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    num_classes: int = 15  # Will be auto-detected from dataset
    
    # Module configurations
    module1: Module1Config = field(default_factory=Module1Config)
    module2: Module2Config = field(default_factory=Module2Config)
    module3: Module3Config = field(default_factory=Module3Config)
    module4: Module4Config = field(default_factory=Module4Config)
    
    # Ablation study settings
    ablation_data_fractions: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])
    ablation_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# Hyperparameter search spaces
HPO_SEARCH_SPACE = {
    'qcnn': {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'conv_layers': [1, 2, 3],
        'n_qubits': [6, 8],  # For 8x8 (6) or 16x16 (8) images
        'encoding_type': ['amplitude', 'angle'],
        'entanglement_pattern': ['ring', 'full'],
    },
    'qnn': {
        'learning_rate': [0.001, 0.005, 0.01],
        'n_layers': [2, 3, 4, 6],
        'n_qubits': [4, 5, 6],
        'optimizer': ['adam', 'rmsprop'],
    },
    'cnn': {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.01],
        'architecture': ['resnet18', 'resnet34', 'mobilenetv2'],
        'freeze_backbone': [True, False],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'scheduler': ['cosine', 'step'],
    },
    'dnn': {
        'hidden_dims': [[512, 256], [256, 128], [512, 256, 128]],
        'dropout': [0.2, 0.3, 0.5],
    }
}


def get_config(module_name: str = "module1") -> ExperimentConfig:
    """Get configuration for specific module"""
    config = ExperimentConfig()
    return config


def save_config(config: ExperimentConfig, path: str):
    """Save configuration to file"""
    import yaml
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError("Config file must be .yaml or .json")


def load_config(path: str) -> ExperimentConfig:
    """Load configuration from file"""
    import yaml
    import json
    
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError("Config file must be .yaml or .json")
    
    # Reconstruct config object (simplified - you might need custom logic)
    return ExperimentConfig(**config_dict)
