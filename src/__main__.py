"""CLI entry point for AlphaEvolve."""

import asyncio
import click
import yaml
import logging
from pathlib import Path

from .alphaevolve import AlphaEvolve, AlphaEvolveConfig


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


@click.group()
def cli():
    """AlphaEvolve - Evolutionary code optimization using Claude AI."""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--problem', '-p', required=True, help='Problem ID to run')
@click.option('--resume', is_flag=True, help='Resume from last checkpoint')
def run(config, problem, resume):
    """Run evolution for a specific problem."""
    # Load configuration
    config_data = {}
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Setup logging
    logging_config = config_data.get('logging', {})
    setup_logging(
        level=logging_config.get('level', 'INFO'),
        log_file=logging_config.get('file')
    )
    
    # Run the appropriate problem
    if problem == "circle_packing":
        from examples.circle_packing.problem import run_circle_packing
        asyncio.run(run_circle_packing(config_path=config))
    elif problem == "function_optimization":
        from examples.function_optimization.problem import run_function_optimization
        asyncio.run(run_function_optimization(config_path=config))
    elif problem == "kissing_spheres":
        from examples.kissing_spheres.problem import run_kissing_spheres
        asyncio.run(run_kissing_spheres(config_path=config))
    else:
        click.echo(f"Unknown problem: {problem}")
        click.echo("Available problems: circle_packing, function_optimization, kissing_spheres")


@cli.command()
@click.option('--output', '-o', default='config/example_config.yaml', help='Output file path')
def generate_config(output):
    """Generate example configuration file."""
    example_config = {
        'evolution': {
            'population_size': 50,
            'generations': 100,
            'elite_size': 5,
            'mutation_rate': 0.8,
            'crossover_rate': 0.2,
            'exploration_rate': 0.1,
            'checkpoint_interval': 10
        },
        'claude': {
            'model': 'claude-sonnet-4-20250514',
            'temperature': 0.7,
            'max_tokens': 4000
        },
        'evaluation': {
            'timeout': 30,
            'memory_limit_mb': 512
        },
        'database': {
            'url': 'sqlite:///data/alphaevolve.db'
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/alphaevolve.log'
        }
    }
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    click.echo(f"Generated example configuration at: {output}")


@cli.command()
def list_problems():
    """List available example problems."""
    problems = [
        ("circle_packing", "Pack 26 circles in a unit square"),
        ("function_optimization", "Minimize the Rastrigin function"),
        ("kissing_spheres", "Find maximum touching spheres in 3D (Newton's kissing number problem)"),
    ]
    
    click.echo("Available problems:")
    for problem_id, description in problems:
        click.echo(f"  {problem_id}: {description}")


if __name__ == "__main__":
    cli()