"""Data loader module for reading markdown files."""

import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DataLoader:
    """Load markdown files from data directory."""
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize with data directory path.
        
        Args:
            data_dir: Path to directory containing markdown files
            
        Raises:
            ValueError: If data_dir does not exist
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        self.data_dir = data_dir
        logger.info(f"DataLoader initialized with directory: {data_dir}")
    
    def load_all_markdown_files(self) -> List[Tuple[str, str]]:
        """
        Scan data directory and load all .md files.
        
        Returns:
            List of (filename, content) tuples
            
        Raises:
            ValueError: If data_dir does not exist
        """
        results = []
        markdown_files = self._scan_directory()
        
        logger.info(f"Found {len(markdown_files)} markdown files in {self.data_dir}")
        
        for filepath in markdown_files:
            try:
                content = self.load_single_file(filepath)
                results.append((os.path.basename(filepath), content))
                logger.info(f"Successfully loaded: {os.path.basename(filepath)} ({len(content)} characters)")
            except (FileNotFoundError, UnicodeDecodeError) as e:
                logger.error(f"Failed to load {filepath}: {e}")
                continue  # Continue with remaining files
        
        logger.info(f"Loaded {len(results)} markdown files successfully")
        return results
    
    def load_single_file(self, filepath: str) -> str:
        """
        Load a single markdown file.
        
        Args:
            filepath: Path to markdown file
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file does not exist
            UnicodeDecodeError: If file encoding is not UTF-8
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {filepath}: {e}")
            raise
    
    def _scan_directory(self) -> List[str]:
        """
        Scan directory for markdown files.
        
        Returns:
            List of full file paths to .md files
        """
        markdown_files = []
        
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.md'):
                full_path = os.path.join(self.data_dir, filename)
                if os.path.isfile(full_path):
                    markdown_files.append(full_path)
        
        return sorted(markdown_files)
