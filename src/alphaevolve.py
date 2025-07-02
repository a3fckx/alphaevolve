"""Main AlphaEvolve orchestrator."""

import asyncio
import logging
import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import yaml

from .database import Database, Program, EvolutionRun, RunStatus, ProgramStatus
from .openai_client import OpenAIClient
from .prompt_sampler import PromptSampler
from .evaluator import CodeEvaluator, EvaluationResult
from .utils import parse_evolve_blocks, reconstruct_code

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlphaEvolveConfig:
    """Configuration for AlphaEvolve."""
    iterations: int
    checkpoint_interval: int

    # LLM settings
    model: str
    temperature: float
    max_tokens: int

    # Evaluation settings
    evaluation_timeout: int
    memory_limit_mb: int


class AlphaEvolve:
    """Main orchestrator for the AlphaEvolve process."""

    def __init__(self,
                 problem_id: str,
                 problem_description: str,
                 evaluation_criteria: str,
                 initial_code: str,
                 problem_type: str,
                 test_cases: List[Dict[str, Any]],
                 custom_evaluator: Optional[Callable] = None,
                 config: Optional[AlphaEvolveConfig] = None,
                 database_url: Optional[str] = None,
                 config_path: str = "config/config.yaml"):
        """Initialize AlphaEvolve."""
        self.problem_id = problem_id
        self.problem_description = problem_description
        self.evaluation_criteria = evaluation_criteria
        self.initial_code = initial_code
        self.problem_type = problem_type
        self.test_cases = test_cases

        # Load configuration from file if not provided
        if config is None:
            config_dict = {}
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

            openai_config = config_dict.get('openai', {})
            evaluation_config = config_dict.get('evaluation', {})

            self.config = AlphaEvolveConfig(
                iterations=config_dict.get('iterations', 100),
                checkpoint_interval=config_dict.get('checkpoint_interval', 10),
                model=openai_config.get('model', 'gemini-2.5-flash-lite-preview-06-17'),
                temperature=openai_config.get('temperature', 0.7),
                max_tokens=openai_config.get('max_tokens', 4096),
                evaluation_timeout=evaluation_config.get('timeout', 30),
                memory_limit_mb=evaluation_config.get('memory_limit_mb', 512),
            )
        else:
            self.config = config

        # Initialize components
        self.db = Database(database_url or os.getenv("DATABASE_URL", "sqlite:///alphaevolve.db"))
        self.llm_client = OpenAIClient(model=self.config.gemini.model)
        self.prompt_sampler = PromptSampler(problem_description, evaluation_criteria)
        self.evaluator = CodeEvaluator(
            timeout=self.config.evaluation_timeout,
            memory_limit_mb=self.config.memory_limit_mb,
            custom_evaluator=custom_evaluator
        )

        # State for block-based evolution
        self.current_run: Optional[EvolutionRun] = None
        self.current_run_id: Optional[str] = None
        self.best_program: Optional[Program] = None
        self.code_template: Optional[str] = None
        self.evolvable_blocks: List[str] = []
        self.block_evolution_idx = 0

    async def run(self, resume_from_checkpoint: bool = False) -> Program:
        """Run the block-based evolution process."""
        try:
            await self._initialize_run(resume_from_checkpoint)
            logger.info(f"Starting evolution run {self.current_run_id} for problem {self.problem_id}")

            for i in range(self.config.iterations):
                logger.info(f"Iteration {i + 1}/{self.config.iterations}")

                # 1. Select a block to evolve
                block_to_evolve_idx = self.block_evolution_idx % len(self.evolvable_blocks)
                
                # 2. Generate a prompt to improve the block
                prompt = self.prompt_sampler.generate_block_evolution_prompt(
                    code_template=self.code_template,
                    evolve_block_index=block_to_evolve_idx,
                    evolve_block_code=self.evolvable_blocks[block_to_evolve_idx]
                )

                # 3. Get a new block from the LLM
                logger.info(f"Evolving block {block_to_evolve_idx}...")
                response = await self.llm_client.generate(
                    user_prompt=prompt.user_prompt,
                    system_prompt=prompt.system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

                if not response:
                    logger.warning("LLM failed to generate a response for the block. Skipping iteration.")
                    continue

                new_block_code = self.llm_client.extract_code(response)
                if not new_block_code:
                    logger.warning("No code extracted from LLM response. Skipping iteration.")
                    continue

                # 4. Create a candidate program
                candidate_blocks = self.evolvable_blocks[:]
                candidate_blocks[block_to_evolve_idx] = new_block_code
                candidate_code = reconstruct_code(self.code_template, candidate_blocks)

                # 5. Evaluate the candidate program
                logger.info("Evaluating candidate program...")
                evaluation_result = await self.evaluator.evaluate(candidate_code, self.test_cases)

                # 6. Save the candidate program and its evaluation
                session = self.db.get_session()
                try:
                    candidate_program = self.db.create_program(
                        session,
                        code=candidate_code,
                        problem_id=self.problem_id,
                        generation=0  # Placeholder, adjust if needed
                    )
                    self.db.update_program_evaluation(
                        session,
                        candidate_program.id,
                        score=evaluation_result.fitness,
                        metrics=evaluation_result.details,
                        status="evaluated",
                        error_log=None
                    )
                finally:
                    session.close()

                # 7. Update the best program if the candidate is better
                if evaluation_result.fitness is not None and (self.best_program.score is None or evaluation_result.fitness > self.best_program.score):
                    logger.info(f"New best program found with score {evaluation_result.fitness} (previously {self.best_program.score if self.best_program.score is not None else 'None'})")
                    self.best_program = candidate_program
                    self.evolvable_blocks = candidate_blocks  # Update the blocks with the improved version
                else:
                    logger.info(f"Candidate program did not improve score. Sticking with current best (score: {self.best_program.score if self.best_program.score is not None else 'None'}).")

                # 8. Update block selection index for next iteration
                self.block_evolution_idx += 1
                
                # 9. Checkpoint
                if (i + 1) % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint()

            # Update run status to completed
            session = self.db.get_session()
            try:
                self.db.update_evolution_run(
                    session,
                    self.current_run_id,
                    status="completed"
                )
            finally:
                session.close()
            logger.info(f"Evolution run {self.current_run_id} completed.")

        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            if self.current_run_id:
                session = self.db.get_session()
                try:
                    self.db.update_evolution_run(
                        session,
                        self.current_run_id,
                        status=RunStatus.PAUSED
                    )
                finally:
                    session.close()
            raise
        
        return self.best_program

    async def _initialize_run(self, resume: bool):
        """Initialize a new evolution run or resume an existing one."""
        if resume:
            # TODO: Implement resume logic
            raise NotImplementedError("Resuming from checkpoint is not yet implemented.")
        
        # Create a new run
        self.current_run = await self.db.create_run(
            problem_id=self.problem_id,
            config=json.dumps(self.config.__dict__),
            status=RunStatus.RUNNING
        )
        self.current_run_id = self.current_run.id

        # Parse the initial code into a template and evolvable blocks
        self.code_template, self.evolvable_blocks = parse_evolve_blocks(self.initial_code)
        
        # Parse the initial code into a template and evolvable blocks
        self.code_template, self.evolvable_blocks = parse_evolve_blocks(self.initial_code)
        
        # Debug: Print the template and blocks to verify parsing
        logger.info("Debug: Code template after parsing:")
        logger.info(self.code_template)
        logger.info("Debug: Evolvable blocks:")
        for i, block in enumerate(self.evolvable_blocks):
            logger.info(f"Block {i}:")
            logger.info(block)
        
        # Reconstruct the initial code for evaluation
        initial_program_code = reconstruct_code(self.code_template, self.evolvable_blocks)
        
        # Debug: Print the reconstructed code before evaluation
        logger.info("Debug: Reconstructed initial code for evaluation:")
        logger.info(initial_program_code)
        
        # Evaluate the initial code to get a baseline
        logger.info("Evaluating initial code...")
        initial_eval = await self.evaluator.evaluate(initial_program_code, self.test_cases)
        
        if initial_eval.fitness is None:
            raise ValueError(f"Initial code failed to evaluate. Error: {initial_eval.error}")

        # Save the initial program as the first "best program"
        session = self.db.get_session()
        try:
            self.best_program = self.db.create_program(
                session,
                code=self.initial_code,
                problem_id=self.problem_id,
                generation=0
            )
            self.db.update_program_evaluation(
                session,
                self.best_program.id,
                score=initial_eval.fitness,
                metrics=initial_eval.details,
                status="evaluated",
                error_log=None
            )
        finally:
            session.close()
            logger.info(f"Initial program evaluated with score: {self.best_program.score}")

    async def _save_checkpoint(self):
        """Save the current state of the evolution run."""
        logger.info(f"Saving checkpoint for run {self.current_run_id}...")
        checkpoint_data = {
            "best_program_id": self.best_program.id,
            "block_evolution_idx": self.block_evolution_idx,
            "code_template": self.code_template,
            "evolvable_blocks": self.evolvable_blocks
        }
        await self.db.update_run_checkpoint(self.current_run_id, json.dumps(checkpoint_data))
        logger.info("Checkpoint saved.")


import asyncio
import logging
import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import yaml

from .database import Database, Program, EvolutionRun, RunStatus, ProgramStatus
from .openai_client import OpenAIClient
from .prompt_sampler import PromptSampler
from .evaluator import CodeEvaluator, EvaluationResult
from .utils import parse_evolve_blocks, reconstruct_code

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlphaEvolveConfig:
    """Configuration for AlphaEvolve."""
    iterations: int
    checkpoint_interval: int
    population_size: int
    elite_size: int
    
    # LLM settings
    model: str
    temperature: float
    max_tokens: int
    
    # Evaluation settings
    evaluation_timeout: int
    memory_limit_mb: int


class AlphaEvolve:
    """Main orchestrator for the AlphaEvolve process."""

    def __init__(self,
                 problem_id: str,
                 problem_description: str,
                 evaluation_criteria: str,
                 initial_code: str,
                 problem_type: str,
                 test_cases: List[Dict[str, Any]],
                 custom_evaluator: Optional[Callable] = None,
                 config: Optional[AlphaEvolveConfig] = None,
                 database_url: Optional[str] = None,
                 config_path: str = "config/config.yaml"):
        """Initialize AlphaEvolve."""
        self.problem_id = problem_id
        self.problem_description = problem_description
        self.evaluation_criteria = evaluation_criteria
        self.initial_code = initial_code
        self.problem_type = problem_type
        self.test_cases = test_cases

        # Load configuration from file if not provided
        if config is None:
            config_dict = {}
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

            openai_config = config_dict.get('openai', {})
            evaluation_config = config_dict.get('evaluation', {})

            self.config = AlphaEvolveConfig(
                iterations=config_dict.get('iterations', 100),
                checkpoint_interval=config_dict.get('checkpoint_interval', 10),
                population_size=config_dict.get('population_size', 10),
                elite_size=config_dict.get('elite_size', 2),
                model=openai_config.get('model', 'gemini-2.5-flash-lite-preview-06-17'),
                temperature=openai_config.get('temperature', 0.7),
                max_tokens=openai_config.get('max_tokens', 4096),
                evaluation_timeout=evaluation_config.get('timeout', 30),
                memory_limit_mb=evaluation_config.get('memory_limit_mb', 512),
            )
        else:
            self.config = config

        # Initialize components
        self.db = Database(database_url or os.getenv("DATABASE_URL", "sqlite:///alphaevolve.db"))
        self.llm_client = OpenAIClient(model=self.config.model)
        self.prompt_sampler = PromptSampler(problem_description, evaluation_criteria)
        self.evaluator = CodeEvaluator(
            timeout=self.config.evaluation_timeout,
            memory_limit_mb=self.config.memory_limit_mb,
            custom_evaluator=custom_evaluator
        )

        # State for block-based evolution
        self.current_run: Optional[EvolutionRun] = None
        self.current_run_id: Optional[str] = None
        self.best_program: Optional[Program] = None
        self.code_template: Optional[str] = None
        self.evolvable_blocks: List[str] = []
        self.block_evolution_idx = 0
        self.population: List[Program] = []
        self.generation: int = 0
        self.iteration: int = 0

    async def run(self, resume_from_checkpoint: bool = False) -> Program:
        """Run the block-based evolution process."""
        try:
            await self._initialize_run(resume_from_checkpoint)
            logger.info(f"Starting evolution run {self.current_run_id} for problem {self.problem_id}")

            for i in range(self.config.iterations):
                logger.info(f"Iteration {i + 1}/{self.config.iterations}")

                # 1. Select a block to evolve
                block_to_evolve_idx = self.block_evolution_idx % len(self.evolvable_blocks)
                
                # 2. Generate a prompt to improve the block
                prompt = self.prompt_sampler.generate_block_evolution_prompt(
                    code_template=self.code_template,
                    evolve_block_index=block_to_evolve_idx,
                    evolve_block_code=self.evolvable_blocks[block_to_evolve_idx]
                )

                # 3. Get a new block from the LLM
                logger.info(f"Evolving block {block_to_evolve_idx}...")
                response = await self.llm_client.generate(
                    user_prompt=prompt.user_prompt,
                    system_prompt=prompt.system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

                if not response:
                    logger.warning("LLM failed to generate a response for the block. Skipping iteration.")
                    continue

                new_block_code = self.llm_client.extract_code(response)
                if not new_block_code:
                    logger.warning("No code extracted from LLM response. Skipping iteration.")
                    continue

                # 4. Create a candidate program
                candidate_blocks = self.evolvable_blocks[:]
                candidate_blocks[block_to_evolve_idx] = new_block_code
                candidate_code = reconstruct_code(self.code_template, candidate_blocks)

                # 5. Evaluate the candidate program
                logger.info("Evaluating candidate program...")
                evaluation_result = await self.evaluator.evaluate(candidate_code, self.test_cases)

                # 6. Save the candidate program and its evaluation
                session = self.db.get_session()
                try:
                    candidate_program = self.db.create_program(
                        session,
                        code=candidate_code,
                        problem_id=self.problem_id,
                        generation=0  # Placeholder, adjust if needed
                    )
                    self.db.update_program_evaluation(
                        session,
                        candidate_program.id,
                        score=evaluation_result.fitness,
                        metrics=evaluation_result.details,
                        status="evaluated",
                        error_log=None
                    )
                finally:
                    session.close()

                # 7. Update the best program if the candidate is better
                if evaluation_result.fitness is not None and (self.best_program.score is None or evaluation_result.fitness > self.best_program.score):
                    logger.info(f"New best program found with score {evaluation_result.fitness} (previously {self.best_program.score if self.best_program.score is not None else 'None'})")
                    self.best_program = candidate_program
                    self.evolvable_blocks = candidate_blocks  # Update the blocks with the improved version
                else:
                    logger.info(f"Candidate program did not improve score. Sticking with current best (score: {self.best_program.score if self.best_program.score is not None else 'None'}).")

                # 8. Update block selection index for next iteration
                self.block_evolution_idx += 1
                
                # 9. Checkpoint
                if (i + 1) % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint()

            # Update run status to completed
            session = self.db.get_session()
            try:
                self.db.update_evolution_run(
                    session,
                    self.current_run_id,
                    status=RunStatus.COMPLETED
                )
            finally:
                session.close()
            logger.info(f"Evolution run {self.current_run_id} completed.")

        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            if self.current_run_id:
                session = self.db.get_session()
                try:
                    self.db.update_evolution_run(
                        session,
                        self.current_run_id,
                        status=RunStatus.PAUSED
                    )
                finally:
                    session.close()
            raise
        
        return self.best_program

    async def _initialize_run(self, resume: bool):
        """Initialize a new evolution run or resume an existing one."""
        session = self.db.get_session()
        try:
            if resume:
                # TODO: Implement resume logic
                raise NotImplementedError("Resuming from checkpoint is not yet implemented.")
            
            # Create a new run
            self.current_run = self.db.create_evolution_run(
                session,
                problem_id=self.problem_id,
                config=self.config.__dict__
            )
            self.current_run_id = self.current_run.id

            # Parse the initial code into a template and evolvable blocks
            self.code_template, self.evolvable_blocks = parse_evolve_blocks(self.initial_code)
            if not self.evolvable_blocks:
                raise ValueError("No evolvable blocks found in the initial code. Use # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END markers.")
            
            logger.info(f"Found {len(self.evolvable_blocks)} evolvable blocks.")
            
            # Reconstruct the initial code for evaluation
            initial_program_code = reconstruct_code(self.code_template, self.evolvable_blocks)
            
            # Debug: Print the reconstructed code before evaluation
            logger.info("Debug: Reconstructed initial code for evaluation:")
            logger.info(initial_program_code)
            
            # Evaluate the initial code to get a baseline
            logger.info("Evaluating initial program...")
            initial_eval = await self.evaluator.evaluate(initial_program_code, self.test_cases)
            
            if initial_eval.fitness is None:
                raise ValueError(f"Initial code failed to evaluate. Error: {initial_eval.error}")

            # Save the initial program as the first "best program"
            self.best_program = self.db.create_program(
                session,
                code=self.initial_code,
                problem_id=self.problem_id,
                generation=0
            )
            self.db.update_program_evaluation(
                session,
                self.best_program.id,
                score=initial_eval.fitness,
                metrics=initial_eval.details,
                status="evaluated",
                error_log=None
            )
            logger.info(f"Initial program evaluated with score: {self.best_program.score}")
        finally:
            session.close()

    async def _save_checkpoint(self):
        """Save the current state of the evolution run."""
        logger.info(f"Saving checkpoint for run {self.current_run_id}...")
        checkpoint_data = {
            "best_program_id": self.best_program.id,
            "block_evolution_idx": self.block_evolution_idx,
            "code_template": self.code_template,
            "evolvable_blocks": self.evolvable_blocks
        }
        session = self.db.get_session()
        try:
            self.db.create_checkpoint(
                session,
                run_id=self.current_run_id,
                generation=self.iteration,
                population_ids=[self.best_program.id] if self.best_program else []
            )
        finally:
            session.close()
        logger.info("Checkpoint saved.")

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
            logger.info(f"Program {program.id} evaluation complete: fitness={result.fitness}, error={result.error}")
            
            updated_program = self.db.update_program_evaluation(
                session,
                program.id,
                score=result.fitness,
                metrics=result.details,
                status="evaluated" if result.error is None and result.fitness is not None else "failed",
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


async def main():
    """Example usage."""
    pass


if __name__ == "__main__":
    asyncio.run(main())
