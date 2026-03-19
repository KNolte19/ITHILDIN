"""
Configuration loader module for ITHILDIN wing analysis pipeline.

This module provides a unified configuration system that can dynamically load
configurations for different insect families (mosquito, drosophila, tsetse).

Available Families:
    - mosquito: Mosquito wing analysis with CNN classification
    - drosophila: Drosophila wing analysis (no CNN classification)
    - tsetse: Tsetse fly wing analysis (no CNN classification)
"""

import os

# Determine root path: use environment variable if set, otherwise use current directory
_DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.environ.get("ITHILDIN_ROOT", _DEFAULT_ROOT)

# ========== Mosquito Configuration ==========
MOSQUITO_CONFIG = { 
    "family": "mosquito",
    "has_classification": True,
    "root_path": _ROOT_PATH,
    "device": "cpu",
    "segmentation_image_size": (640, 320),
    "landmark_image_size": (480, 240),
    "classifier_image_size": (480, 240),
    "model_paths": {
        "segmentation": "training/models/segmentation_weights_fold-1.pth",
        "landmark": "training/models/landmark_weights_fold-1_development.pth",
        "classification": "training/models/classifier_1_evaluation.pth"
    },
    "N_landmarks": 17,
    "N_semilandmarks": 52,
    "index_most_left_landmark": 0,
    "index_most_right_landmark": 5,
    "index_most_upper_landmark": 1,
    "index_most_lower_landmark": 8,
    "allowed_connections": [
        (0, 1), (0, 14),       
        (1, 2),                 
        (2, 3), (2, 15),        
        (3, 4), (3, 15),       
        (4, 5), (4, 13),       
        (5, 6), (5, 16),    
        (6, 7), (6, 16),  
        (7, 8), (7, 10), 
        (8, 9),             
        (9, 10),         
        (12, 16),         
        (14, 15),           
    ],
    "not_allowed_connections": [
        (0, 2), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13),
        (1, 3), (1, 14), (1, 13),
        (2, 4), (2, 12), (2, 13), (2, 14),
        (3, 5), (3, 13), (3, 14),
        (4, 6), (4, 12), (4, 14), (4, 15), (4, 16),
        (5, 7), (5, 12), (5, 13), 
        (6, 8), (6, 10), (6, 12), (6, 13),
        (7, 9), (7, 11), (7, 12), (7, 16),
        (8, 10),
        (9, 11), (9, 12), (9, 13), (9, 14),
        (10, 12), (10, 13), (10, 16),
        (11, 13), (11, 14), (11, 15), (11, 16),
        (12, 15),
        (13, 15), (13, 16),
        (14, 16)
    ],
    "semilandmarks_per_connection": [(4), (4), (1), (1), (4), (1), (4), (1), (4), (1), (4), (1), (4), (1), (4), (4), (1), (4), (4),],
    "classifier_species_list": [
        'Aedes_aegypti', 'Aedes_albopictus', 'Aedes_annulipes-group',
        'Aedes_caspius', 'Aedes_cinereus-geminus-pair',
        'Aedes_communis-punctor-pair', 'Aedes_detritus-coluzzi-pair',
        'Aedes_japonicus', 'Aedes_koreicus', 'Aedes_rusticus',
        'Aedes_scapularis', 'Aedes_sticticus', 'Aedes_vexans',
        'Anopheles_aquasalis', 'Anopheles_claviger-petragnani-pair',
        'Anopheles_coluzzii', 'Anopheles_coustani-group',
        'Anopheles_darlingi', 'Anopheles_maculipennis s.l.',
        'Anopheles_moucheti', 'Anopheles_paludis', 'Anopheles_stephensi',
        'Coquillettidia_richiardii', 'Culex_modestus',
        'Culex_pipiens s.l.-torrentium-pair', 'Culex_vishnui-group',
        'Culiseta_annulata-subochrea-pair',
        'Culiseta_morsitans-fumipennis-pair', 'Haemagogus_janthinomys',
        'Haemagogus_leucocelaenus', 'Limatus_durhamii',
        'Sabethes_albiprivus', 'Wyeomyia_arthrostigma', 'other'
    ],
    "landmark_reference_path": "analysis/LDA_reference_dataframe_landmarks.csv",
    "semilandmark_reference_path": "analysis/LDA_reference_dataframe_semilandmarks.csv",
}

# ========== Drosophila Configuration ==========
DROSOPHILA_CONFIG = { 
    "family": "drosophila",
    "has_classification": False,
    "root_path": _ROOT_PATH,
    "device": "cpu",
    "segmentation_image_size": (640, 320),
    "landmark_image_size": (480, 240),
    "classifier_image_size": (480, 240),
    "model_paths": {
        "segmentation": "training/models_droso/drosophila_segmentation_weights_fold-0.pth",
        "landmark": "training/models_droso/landmark_weights_fold-0_Drosophila.pth",
        "classification": "training/models/classifier_1_evaluation.pth"
    },
    "N_landmarks": 14,
    "N_semilandmarks": 26,
    "index_most_left_landmark": 3,
    "index_most_right_landmark": 6,
    "index_most_upper_landmark": 0,
    "index_most_lower_landmark": 10,
    "allowed_connections": [
        (0, 3), (0, 9), (0, 11),     
        (1, 2), (1, 5),              
        (2, 9),                      
        (4, 12),                     
        (5, 6),                      
        (6, 7), (6, 10),            
        (7, 12), (7, 13),          
        (8, 13),                     
        (10, 13)                     
    ],
    "not_allowed_connections": [],
    "semilandmarks_per_connection": [
        1, 4, 1, 1, 4, 4, 1, 1, 2, 3, 1, 1, 1, 1
    ],
    "classifier_species_list": [],
    "landmark_reference_path": "analysis/LDA_reference_dataframe_landmarks.csv",
    "semilandmark_reference_path": "analysis/LDA_reference_dataframe_semilandmarks.csv",
}

# ========== Tsetse Configuration ==========
TSETSE_CONFIG = {
    "family": "tsetse",
    "has_classification": False,
    "root_path": _ROOT_PATH,
    "device": "cpu",
    "segmentation_image_size": (640, 320),
    "landmark_image_size": (480, 240),
    "classifier_image_size": (480, 240),
    "model_paths": {
        "segmentation": "training/models_tsetse/tsetse_segmentation_weights_fold-0.pth",
        "landmark": "training/models_tsetse/tsetse_landmark_weights_fold-0.pth",
        "classification": "training/models/classifier_1_evaluation.pth",
    },
    "N_landmarks": 11,
    "N_semilandmarks": 12,
    "index_most_left_landmark": 5,
    "index_most_right_landmark": 0,
    "index_most_upper_landmark": 1,
    "index_most_lower_landmark": 7,
    "allowed_connections": [
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (6, 9),
        (7, 10),
        (9, 10),
    ],
    "not_allowed_connections": [],
    "semilandmarks_per_connection": [3, 1, 3, 3, 2, 2, 5, 3],
    "classifier_species_list": [],
    "landmark_reference_path": "analysis/LDA_reference_dataframe_landmarks.csv",
    "semilandmark_reference_path": "analysis/LDA_reference_dataframe_semilandmarks.csv",
}

# Available insect families
AVAILABLE_FAMILIES = {
    "mosquito": MOSQUITO_CONFIG,
    "drosophila": DROSOPHILA_CONFIG,
    "tsetse": TSETSE_CONFIG,
}

# Default configuration
_current_config = MOSQUITO_CONFIG


def get_config(family="mosquito"):
    """
    Get configuration for specified insect family.
    
    Args:
        family (str): Insect family name ('mosquito', 'drosophila', or 'tsetse')
    
    Returns:
        dict: Configuration dictionary for the specified family
    
    Raises:
        ValueError: If family name is not recognized
    """
    if family not in AVAILABLE_FAMILIES:
        raise ValueError(
            f"Unknown family '{family}'. Available families: {list(AVAILABLE_FAMILIES.keys())}"
        )
    return AVAILABLE_FAMILIES[family]


def set_config(family="mosquito"):
    """
    Set the current global configuration to specified insect family.
    
    Args:
        family (str): Insect family name ('mosquito', 'drosophila', or 'tsetse')
    
    Returns:
        dict: The new current configuration
    
    Raises:
        ValueError: If family name is not recognized
    """
    global _current_config
    _current_config = get_config(family)
    return _current_config


def get_current_config():
    """
    Get the currently active configuration.
    
    Returns:
        dict: Current configuration dictionary
    """
    return _current_config
