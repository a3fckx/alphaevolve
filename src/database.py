"""Database models and operations for AlphaEvolve."""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

Base = declarative_base()


class ProgramStatus(str, Enum):
    """Status of a program in the evolution process."""
    PENDING = "pending"
    EVALUATED = "evaluated"
    FAILED = "failed"


class RunStatus(str, Enum):
    """Status of an evolution run."""
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"


class Program(Base):
    """Represents a code program in the evolution process."""
    __tablename__ = "programs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(Text, nullable=False)
    score = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)  # Detailed performance metrics
    generation = Column(Integer, nullable=False)
    parent_id = Column(String(36), ForeignKey("programs.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    problem_id = Column(String(100), nullable=False)
    status = Column(SQLEnum(ProgramStatus), default=ProgramStatus.PENDING)
    error_log = Column(Text, nullable=True)
    
    # Relationships
    parent = relationship("Program", remote_side=[id], backref="children")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert program to dictionary."""
        return {
            "id": self.id,
            "code": self.code,
            "score": self.score,
            "metrics": self.metrics,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "problem_id": self.problem_id,
            "status": self.status.value if self.status else None,
            "error_log": self.error_log
        }


class EvolutionRun(Base):
    """Represents an evolution run."""
    __tablename__ = "evolution_runs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    problem_id = Column(String(100), nullable=False)
    config = Column(JSON, nullable=False)  # Evolution parameters
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    best_score = Column(Float, nullable=True)
    best_program_id = Column(String(36), ForeignKey("programs.id"), nullable=True)
    generation_count = Column(Integer, default=0)
    status = Column(SQLEnum(RunStatus), default=RunStatus.RUNNING)
    
    # Relationships
    best_program = relationship("Program", foreign_keys=[best_program_id])
    checkpoints = relationship("Checkpoint", back_populates="run", cascade="all, delete-orphan")


class Checkpoint(Base):
    """Represents a checkpoint in the evolution process."""
    __tablename__ = "checkpoints"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey("evolution_runs.id"), nullable=False)
    generation = Column(Integer, nullable=False)
    population_snapshot = Column(JSON, nullable=False)  # IDs and scores of current population
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("EvolutionRun", back_populates="checkpoints")


class Database:
    """Database interface for AlphaEvolve."""
    
    def __init__(self, database_url: str = "sqlite:///alphaevolve.db"):
        """Initialize database connection."""
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def create_program(self, session: Session, code: str, problem_id: str, 
                      generation: int, parent_id: Optional[str] = None) -> Program:
        """Create a new program."""
        program = Program(
            code=code,
            problem_id=problem_id,
            generation=generation,
            parent_id=parent_id
        )
        session.add(program)
        session.commit()
        session.refresh(program)
        return program
    
    def update_program_evaluation(self, session: Session, program_id: str, 
                                 score: float, metrics: Dict[str, Any], 
                                 status: str = "evaluated",
                                 error_log: Optional[str] = None) -> Program:
        """Update program with evaluation results."""
        program = session.query(Program).filter_by(id=program_id).first()
        if program:
            program.score = score
            program.metrics = metrics
            program.status = ProgramStatus(status)
            program.error_log = error_log
            session.commit()
            session.refresh(program)
        return program
    
    def get_top_programs(self, session: Session, problem_id: str, 
                        limit: int = 10, generation: Optional[int] = None) -> List[Program]:
        """Get top performing programs."""
        query = session.query(Program).filter_by(
            problem_id=problem_id,
            status=ProgramStatus.EVALUATED
        )
        
        if generation is not None:
            query = query.filter_by(generation=generation)
        
        return query.order_by(Program.score.desc()).limit(limit).all()
    
    def create_evolution_run(self, session: Session, problem_id: str, 
                           config: Dict[str, Any]) -> EvolutionRun:
        """Create a new evolution run."""
        run = EvolutionRun(
            problem_id=problem_id,
            config=config
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    
    def update_evolution_run(self, session: Session, run_id: str, 
                           best_score: Optional[float] = None,
                           best_program_id: Optional[str] = None,
                           generation_count: Optional[int] = None,
                           status: Optional[RunStatus] = None) -> EvolutionRun:
        """Update evolution run progress."""
        run = session.query(EvolutionRun).filter_by(id=run_id).first()
        if run:
            if best_score is not None:
                run.best_score = best_score
            if best_program_id is not None:
                run.best_program_id = best_program_id
            if generation_count is not None:
                run.generation_count = generation_count
            if status is not None:
                run.status = status
                if status == RunStatus.COMPLETED:
                    run.end_time = datetime.utcnow()
            session.commit()
            session.refresh(run)
        return run
    
    def create_checkpoint(self, session: Session, run_id: str, 
                         generation: int, population_ids: List[str]) -> Checkpoint:
        """Create a checkpoint."""
        # Get current population with scores
        programs = session.query(Program).filter(Program.id.in_(population_ids)).all()
        population_snapshot = [
            {"id": p.id, "score": p.score, "metrics": p.metrics}
            for p in programs
        ]
        
        checkpoint = Checkpoint(
            run_id=run_id,
            generation=generation,
            population_snapshot=population_snapshot
        )
        session.add(checkpoint)
        session.commit()
        session.refresh(checkpoint)
        return checkpoint
    
    def get_latest_checkpoint(self, session: Session, run_id: str) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a run."""
        return session.query(Checkpoint).filter_by(
            run_id=run_id
        ).order_by(Checkpoint.generation.desc()).first()
    
    def get_programs_by_ids(self, session: Session, program_ids: List[str]) -> List[Program]:
        """Get programs by their IDs."""
        return session.query(Program).filter(Program.id.in_(program_ids)).all()
    
    def get_program_by_id(self, session: Session, program_id: str) -> Optional[Program]:
        """Get a single program by ID."""
        return session.query(Program).filter(Program.id == program_id).first()