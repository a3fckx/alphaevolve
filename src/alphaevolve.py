"""Main AlphaEvolve orchestrator."""

import asyncio
import logging
import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

from .database import Database, Program, EvolutionRun, RunStatus, ProgramStatus
from .openai_client import OpenAIClient
from .prompt_sampler import PromptSampler
from .evaluator import CodeEvaluator, EvaluationResult
from .evolution_strategies import EvolutionStrategy
import yaml

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlphaEvolveConfig:
    """Configuration for AlphaEvolve."""
    population_size: int
    generations: int
    elite_size: int
    mutation_rate: float
    crossover_rate: float
    exploration_rate: float
    
    # Gemini API settings
    model: str
    temperature: float
    max_tokens: int
    
    # Evaluation settings
    evaluation_timeout: int
    memory_limit_mb: int
    
    # Checkpointing
    checkpoint_interval: int
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AlphaEvolveConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class AlphaEvolve:
    """Main orchestrator for evolutionary code optimization."""
    
    def __init__(self, 
                 problem_id: str,
                 problem_description: str,
                 evaluation_criteria: str,
                 problem_type: str = "optimization",
                 test_cases: Optional[Dict[str, Any]] = None,
                 custom_evaluator: Optional[Callable] = None,
                 config: Optional[AlphaEvolveConfig] = None,
                 database_url: Optional[str] = None,
                 config_path: str = "config/config.yaml"):
        """Initialize AlphaEvolve."""
        self.problem_id = problem_id
        self.problem_description = problem_description
        self.evaluation_criteria = evaluation_criteria
        self.problem_type = problem_type
        self.test_cases = test_cases
        
        # Load configuration from file if not provided
        if config is None:
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                config_dict = {}
                
            evolution_config = config_dict.get('evolution', {})
            openai_config = config_dict.get('openai', {})
            evaluation_config = config_dict.get('evaluation', {})
            self.config = AlphaEvolveConfig(
                population_size=evolution_config.get('population_size', 50),
                generations=evolution_config.get('generations', 100),
                elite_size=evolution_config.get('elite_size', 5),
                mutation_rate=evolution_config.get('mutation_rate', 0.8),
                crossover_rate=evolution_config.get('crossover_rate', 0.2),
                exploration_rate=evolution_config.get('exploration_rate', 0.1),
                model=openai_config.get('model', 'gemini-2.5-flash'),
                temperature=openai_config.get('temperature', 0.7),
                max_tokens=openai_config.get('max_tokens', 4096),
                evaluation_timeout=evaluation_config.get('timeout', 30),
                memory_limit_mb=evaluation_config.get('memory_limit_mb', 512),
                checkpoint_interval=evolution_config.get('checkpoint_interval', 10)
            )
        else:
            self.config = config
        
        # Initialize components
        self.db = Database(database_url or os.getenv("DATABASE_URL", "sqlite:///alphaevolve.db"))
        self.claude = OpenAIClient(model=self.config.model)
        self.prompt_sampler = PromptSampler(problem_description, evaluation_criteria)
        self.evaluator = CodeEvaluator(
            timeout=self.config.evaluation_timeout,
            memory_limit_mb=self.config.memory_limit_mb,
            custom_evaluator=custom_evaluator
        )
        self.evolution_strategy = EvolutionStrategy(
            population_size=self.config.population_size,
            elite_size=self.config.elite_size,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            exploration_rate=self.config.exploration_rate
        )
        
        # State
        self.current_run: Optional[EvolutionRun] = None
        self.current_run_id: Optional[str] = None
        self.population: List[Program] = []
        self.generation = 0
        self.best_program: Optional[Program] = None
    
    async def run(self, resume_from_checkpoint: bool = False) -> Program:
        """Run the evolution process."""
        try:
            # Initialize or resume run
            if resume_from_checkpoint:
                await self._resume_from_checkpoint()
            else:
                await self._initialize_run()
            
            # Main evolution loop
            while self.generation < self.config.generations:
                logger.info(f"Generation {self.generation + 1}/{self.config.generations}")
                
                # Evolve population
                await self._evolve_generation()
                
                # Update statistics
                await self._update_statistics()
                
                # Checkpoint if needed
                if (self.generation + 1) % self.config.checkpoint_interval == 0:
                    await self._create_checkpoint()
                
                self.generation += 1
                
                # Check for early stopping
                if await self._should_stop_early():
                    logger.info("Early stopping criteria met")
                    break
            
            # Finalize run
            await self._finalize_run()
            
            if not self.best_program:
                logger.error("No valid programs found during evolution")
                return None
            
            return self.best_program
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            if self.current_run:
                await self._update_run_status(RunStatus.PAUSED)
            raise
        finally:
            await self.claude.close()
    
    async def _initialize_run(self):
        """Initialize a new evolution run."""
        session = self.db.get_session()
        try:
            # Create run
            self.current_run = self.db.create_evolution_run(
                session, 
                self.problem_id,
                self.config.__dict__
            )
            self.current_run_id = self.current_run.id
            
            # Generate initial population
            logger.info("Generating initial population...")
            initial_prompts = [
                self.prompt_sampler.generate_initial_prompt()
                for _ in range(self.config.population_size)
            ]
            
            all_programs = []
            for i, prompt in enumerate(initial_prompts):
                try:
                    response = await self.claude.generate(
                        prompt.user_prompt,
                        system_prompt=prompt.system_prompt,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    code = self.claude.extract_code(response)
                    # Save generated code for debugging
                    with open(f"data/generated_code_{self.problem_id}_{i}.py", "w") as f:
                        f.write(code)
                    logger.debug(f"Generated code {i}: {code[:200]}...")
                    
                    program = self.db.create_program(
                        session,
                        code=code,
                        problem_id=self.problem_id,
                        generation=0
                    )
                    all_programs.append(program)
                except Exception as e:
                    logger.error(f"Failed to generate program {i}: {e}")
            
            # Evaluate initial population
            logger.info("Evaluating initial population...")
            logger.info(f"Number of programs to evaluate: {len(all_programs)}")
            await self._evaluate_programs(all_programs)
            
            # Filter programs that were successfully evaluated
            self.population = []
            for prog in all_programs:
                logger.debug(f"Program {prog.id}: status={prog.status}, score={prog.score}")
                if prog.status == ProgramStatus.EVALUATED and prog.score is not None:
                    self.population.append(prog)
                    logger.debug(f"Added program {prog.id} to population with score {prog.score}")
            
            self.generation = 0
            
            logger.info(f"Initial population size: {len(self.population)}")
            
        finally:
            session.close()
    
    async def _evolve_generation(self):
        """Evolve one generation."""
        session = self.db.get_session()
        try:
            # Check if population is empty
            if not self.population:
                logger.error("Population is empty, cannot evolve")
                # Try to regenerate initial population
                await self._initialize_run()
                return
            
            new_population = []
            
            # Keep elite
            elite = self.evolution_strategy.select_elite(self.population)
            new_population.extend(elite)
            logger.info(f"Keeping {len(elite)} elite programs")
            
            # Generate new programs
            programs_to_generate = self.config.population_size - len(elite)
            logger.info(f"Generating {programs_to_generate} new programs")
            
            # Batch generation to avoid overwhelming the API
            batch_size = 5  # Process 5 programs at a time
            generated_count = 0
            
            while len(new_population) < self.config.population_size:
                # Prepare batch of prompts
                batch_prompts = []
                batch_tasks = []
                
                for _ in range(min(batch_size, self.config.population_size - len(new_population))):
                    # Select parent and strategy
                    parent = self.evolution_strategy.select_parent(self.population)
                    strategy = self.evolution_strategy.select_evolution_strategy()
                    
                    # Generate evolution prompt
                    prompt = self.prompt_sampler.generate_evolution_prompt(
                        parent=parent,
                        population=self.population,
                        strategy=strategy
                    )
                    
                    # Create task
                    task = self._generate_and_evaluate_program(
                        prompt,
                        parent_id=parent.id,
                        generation=self.generation + 1
                    )
                    batch_tasks.append(task)
                
                # Execute batch concurrently
                logger.info(f"Processing batch of {len(batch_tasks)} programs (total generated: {generated_count})")
                batch_programs = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter successful programs and handle exceptions
                for i, result in enumerate(batch_programs):
                    if isinstance(result, Exception):
                        logger.error(f"Task {i} failed with exception: {type(result).__name__}: {result}")
                    elif result and result.status == ProgramStatus.EVALUATED:
                        new_population.append(result)
                        logger.info(f"Added program {result.id} to new population (score: {result.score})")
                    elif result:
                        logger.warning(f"Program {result.id} failed evaluation")
                
                generated_count += len(batch_tasks)
                
                # Add a small delay between batches to avoid rate limiting
                if len(new_population) < self.config.population_size:
                    await asyncio.sleep(1.0)
            
            # Update population
            self.population = new_population[:self.config.population_size]
            logger.info(f"New population size: {len(self.population)}")
            
            # If population is still too small, log a warning
            if len(self.population) < self.config.elite_size:
                logger.warning(f"Population size ({len(self.population)}) is smaller than elite size ({self.config.elite_size})")
            
        finally:
            session.close()
    
    async def _generate_and_evaluate_program(self, 
                                           prompt, 
                                           parent_id: str,
                                           generation: int) -> Optional[Program]:
        """Generate and evaluate a single program."""
        session = self.db.get_session()
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate code
            logger.info(f"Generating program for generation {generation} with parent {parent_id}")
            logger.debug(f"Prompt (first 200 chars): {prompt.user_prompt[:200]}...")
            
            try:
                response = await asyncio.wait_for(
                    self.claude.generate(
                        prompt.user_prompt,
                        system_prompt=prompt.system_prompt,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    ),
                    timeout=60.0  # 60 second timeout for API call
                )
                logger.info(f"Code generation completed in {asyncio.get_event_loop().time() - start_time:.2f}s")
            except asyncio.TimeoutError:
                logger.error(f"API timeout after 60s for generation {generation}")
                return None
            except Exception as e:
                logger.error(f"API error: {type(e).__name__}: {e}")
                return None
            
            code = self.claude.extract_code(response)
            logger.debug(f"Extracted code length: {len(code)} characters")
            
            # Create program
            program = self.db.create_program(
                session,
                code=code,
                problem_id=self.problem_id,
                generation=generation,
                parent_id=parent_id
            )
            logger.info(f"Created program {program.id}")
            
            # Evaluate
            logger.info(f"Starting evaluation of program {program.id}")
            eval_start = asyncio.get_event_loop().time()
            
            try:
                result = await asyncio.wait_for(
                    self.evaluator.evaluate(
                        code,
                        self.problem_type,
                        self.test_cases
                    ),
                    timeout=self.config.evaluation_timeout + 10  # Add buffer to evaluation timeout
                )
                logger.info(f"Evaluation completed in {asyncio.get_event_loop().time() - eval_start:.2f}s")
                logger.info(f"Evaluation result for {program.id}: success={result.success}, score={result.score}")
            except asyncio.TimeoutError:
                logger.error(f"Evaluation timeout for program {program.id}")
                result = EvaluationResult(
                    score=0.0,
                    metrics={'error': 'Evaluation timeout'},
                    success=False,
                    error="Evaluation timeout"
                )
            
            # Update program with results
            self.db.update_program_evaluation(
                session,
                program.id,
                score=result.score,
                metrics=result.metrics,
                status="evaluated" if result.success else "failed",
                error_log=result.error
            )
            
            # Refresh the program object to get updated values
            session.refresh(program)
            
            total_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Total time for program {program.id}: {total_time:.2f}s")
            
            return program if result.success else None
            
        except Exception as e:
            logger.error(f"Failed to generate/evaluate program: {type(e).__name__}: {e}", exc_info=True)
            return None
        finally:
            session.close()
    
    async def _evaluate_programs(self, programs: List[Program]):
        """Evaluate multiple programs concurrently."""
        tasks = [
            self._evaluate_single_program(program)
            for program in programs
        ]
        await asyncio.gather(*tasks)
    
    async def _evaluate_single_program(self, program: Program):
        """Evaluate a single program."""
        session = self.db.get_session()
        try:
            logger.info(f"Evaluating program {program.id}")
            result = await self.evaluator.evaluate(
                program.code,
                self.problem_type,
                self.test_cases
            )
            logger.info(f"Program {program.id} evaluation complete: success={result.success}, score={result.score}")
            
            updated_program = self.db.update_program_evaluation(
                session,
                program.id,
                score=result.score,
                metrics=result.metrics,
                status="evaluated" if result.success else "failed",
                error_log=result.error
            )
            
            # Update the program object with new values
            if updated_program:
                program.score = updated_program.score
                program.metrics = updated_program.metrics
                program.status = updated_program.status
                program.error_log = updated_program.error_log
            
        except Exception as e:
            logger.error(f"Failed to evaluate program {program.id}: {e}", exc_info=True)
        finally:
            session.close()
    
    async def _update_statistics(self):
        """Update run statistics."""
        if not self.population:
            logger.warning("No programs in population to update statistics")
            return
            
        session = self.db.get_session()
        try:
            # Find best program
            best = max(self.population, key=lambda p: p.score if p.score is not None else float('-inf'))
            
            if not self.best_program or best.score > self.best_program.score:
                self.best_program = best
                logger.info(f"New best score: {best.score:.4f}")
                
                # Update run
                self.db.update_evolution_run(
                    session,
                    self.current_run_id,
                    best_score=best.score,
                    best_program_id=best.id,
                    generation_count=self.generation + 1
                )
            
            # Log generation statistics
            scores = [p.score for p in self.population if p.score is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                logger.info(f"Generation {self.generation + 1} - Avg: {avg_score:.4f}, Best: {max(scores):.4f}")
            
        finally:
            session.close()
    
    async def _create_checkpoint(self):
        """Create a checkpoint."""
        session = self.db.get_session()
        try:
            population_ids = [p.id for p in self.population]
            checkpoint = self.db.create_checkpoint(
                session,
                self.current_run_id,
                self.generation,
                population_ids
            )
            logger.info(f"Created checkpoint at generation {self.generation}")
        finally:
            session.close()
    
    async def _resume_from_checkpoint(self):
        """Resume from the latest checkpoint."""
        session = self.db.get_session()
        try:
            # Find latest run
            latest_run = session.query(EvolutionRun).filter_by(
                problem_id=self.problem_id,
                status=RunStatus.RUNNING
            ).order_by(EvolutionRun.start_time.desc()).first()
            
            if not latest_run:
                logger.info("No resumable run found, starting fresh")
                await self._initialize_run()
                return
            
            self.current_run = latest_run
            
            # Find latest checkpoint
            checkpoint = self.db.get_latest_checkpoint(session, latest_run.id)
            if not checkpoint:
                logger.info("No checkpoint found, starting fresh")
                await self._initialize_run()
                return
            
            # Restore population
            program_ids = [p['id'] for p in checkpoint.population_snapshot]
            self.population = self.db.get_programs_by_ids(session, program_ids)
            self.generation = checkpoint.generation
            
            logger.info(f"Resumed from generation {self.generation}")
            
        finally:
            session.close()
    
    async def _should_stop_early(self) -> bool:
        """Check if we should stop early."""
        # Stop if perfect score achieved
        if self.best_program and self.best_program.score >= 1.0:
            return True
        
        # Stop if no improvement for many generations
        # (would need to track this)
        
        return False
    
    async def _finalize_run(self):
        """Finalize the evolution run and save results."""
        await self._update_run_status(RunStatus.COMPLETED)
        if self.best_program:
            logger.info(f"Evolution completed. Best score: {self.best_program.score:.4f}")
            # Save the best program to a file
            result_filename = f"data/best_program_{self.problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            with open(result_filename, "w") as f:
                f.write(f"# Best Program for {self.problem_id}\n")
                f.write(f"# Score: {self.best_program.score}\n")
                if self.best_program.metrics:
                    f.write(f"# Metrics: {json.dumps(self.best_program.metrics, indent=2)}\n")
                f.write("\n")
                f.write(self.best_program.code)
            logger.info(f"Best program saved to {result_filename}")
        else:
            logger.warning("Evolution completed but no valid programs were found.")
    
    async def _update_run_status(self, status: RunStatus):
        """Update run status."""
        if not self.current_run_id:
            return
        
        session = self.db.get_session()
        try:
            self.db.update_evolution_run(
                session,
                self.current_run_id,
                status=status
            )
        finally:
            session.close()


async def main():
    """Example usage."""
    # This would be called with proper configuration
    pass


if __name__ == "__main__":
    asyncio.run(main())
