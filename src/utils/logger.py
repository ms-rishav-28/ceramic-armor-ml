"""
Logger system with Windows-compatible file rotation and structured logging.

This module provides a centralized logging system with dual output (file and console),
automatic file rotation, and Windows-compatible path handling.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os
from datetime import datetime


class Logger_System:
    """
    Centralized logging system with file rotation and structured output.
    
    Features:
    - Dual output (console and file)
    - Automatic file rotation (10MB limit, 5 backup files)
    - Windows-compatible path handling
    - Structured logging with timestamps and module identification
    - Configurable log levels
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized: bool = False
    _log_dir: Optional[Path] = None
    
    @classmethod
    def initialize(cls, log_dir: Optional[Path] = None, level: str = "INFO") -> None:
        """
        Initialize the logging system with specified directory and level.
        
        Args:
            log_dir: Directory for log files. If None, uses 'logs' in current directory
            level: Default logging level (DEBUG, INFO, WARNING, ERROR)
        """
        if cls._initialized:
            return
            
        # Set up log directory with Windows-compatible paths
        if log_dir is None:
            cls._log_dir = Path.cwd() / "logs"
        else:
            cls._log_dir = Path(log_dir)
            
        # Create log directory if it doesn't exist
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default level
        cls._default_level = getattr(logging, level.upper(), logging.INFO)
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (typically module name)
            level: Log level for this logger (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.initialize()
            
        if name in cls._loggers:
            return cls._loggers[name]
            
        # Create new logger
        logger = logging.getLogger(name)
        
        # Set level
        if level:
            log_level = getattr(logging, level.upper(), cls._default_level)
        else:
            log_level = cls._default_level
        logger.setLevel(log_level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            fmt='%(levelname)s | %(name)s | %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Add file handler with rotation
        if cls._log_dir:
            file_handler = cls._create_file_handler(name, detailed_formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _create_file_handler(cls, logger_name: str, formatter: logging.Formatter) -> logging.Handler:
        """
        Create a rotating file handler for the logger.
        
        Args:
            logger_name: Name of the logger
            formatter: Formatter for log messages
            
        Returns:
            Configured rotating file handler
        """
        # Create log file path with Windows-compatible naming
        safe_name = logger_name.replace('.', '_').replace('/', '_').replace('\\', '_')
        log_file = cls._log_dir / f"{safe_name}.log"
        
        # Create rotating file handler (10MB max, 5 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        return file_handler
    
    @classmethod
    def set_level(cls, logger_name: str, level: str) -> None:
        """
        Set the logging level for a specific logger.
        
        Args:
            logger_name: Name of the logger to modify
            level: New logging level (DEBUG, INFO, WARNING, ERROR)
        """
        if logger_name in cls._loggers:
            log_level = getattr(logging, level.upper(), logging.INFO)
            cls._loggers[logger_name].setLevel(log_level)
    
    @classmethod
    def get_log_directory(cls) -> Optional[Path]:
        """
        Get the current log directory path.
        
        Returns:
            Path to log directory or None if not initialized
        """
        return cls._log_dir


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Convenience function to get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    return Logger_System.get_logger(name, level)


def setup_file_handler(log_dir: Path) -> logging.FileHandler:
    """
    Set up a file handler for logging with Windows-compatible paths.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured file handler
        
    Note:
        This function is deprecated. Use Logger_System.get_logger() instead.
    """
    Logger_System.initialize(log_dir)
    return Logger_System._create_file_handler("default", logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))


# Initialize the logging system when module is imported
Logger_System.initialize()