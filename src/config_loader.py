"""Configuration loader for AlphaEvolve."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from .alphaevolve import AlphaEvolveConfig


def load_config(config_path: Optional[str] = None) -> AlphaEvolveConfig:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        AlphaEvolveConfig instance
    """
    config_data = {}
    
    # Try to load from file
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    elif Path('config/config.yaml').exists():
        # Try default location
        with open('config/config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Extract relevant sections
    evolution_config = config_data.get('evolution', {})
    openai_config = config_data.get('openai', {})
    evaluation_config = config_data.get('evaluation', {})
    
    # Create config object with values from file or defaults
    return AlphaEvolveConfig(
        # Evolution settings
        population_size=evolution_config.get('population_size', 50),
        generations=evolution_config.get('generations', 100),
        elite_size=evolution_config.get('elite_size', 5),
        mutation_rate=evolution_config.get('mutation_rate', 0.8),
        crossover_rate=evolution_config.get('crossover_rate', 0.2),
        exploration_rate=evolution_config.get('exploration_rate', 0.1),
        checkpoint_interval=evolution_config.get('checkpoint_interval', 10),
        
        # OpenAI settings - model MUST come from config
        model=openai_config.get('model', 'google/gemini-2.0-flash-exp:free'),
        temperature=openai_config.get('temperature', 0.7),
        max_tokens=openai_config.get('max_tokens', 20000),
        
        # Evaluation settings
        evaluation_timeout=evaluation_config.get('timeout', 30),
        memory_limit_mb=evaluation_config.get('memory_limit_mb', 512)
    )


def get_database_url(config_data: Dict[str, Any]) -> str:
    """Get database URL from config or environment."""
    db_config = config_data.get('database', {})
    return db_config.get('url', os.getenv('DATABASE_URL', 'sqlite:///data/alphaevolve.db'))
