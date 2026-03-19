"""
Central configuration file for ITHILDIN wing analysis pipeline.

This module provides backward compatibility by maintaining a CONFIG dictionary
that can be dynamically updated based on the selected insect family.

For new code, prefer using config_loader.get_config(family) directly.
"""

from config_loader import get_current_config, set_config, MOSQUITO_CONFIG

# Default CONFIG for backward compatibility (mosquito)
CONFIG = MOSQUITO_CONFIG.copy()

def update_config(family="mosquito"):
    """
    Update the global CONFIG dictionary to match the specified family.
    
    Args:
        family (str): Insect family name
    """
    global CONFIG
    new_config = set_config(family)
    CONFIG.clear()
    CONFIG.update(new_config)