"""
Embedded ML - Convert PyTorch models to C++/Rust for embedded platforms
"""

__version__ = "0.1.0"

from .parser.model_parser import ModelParser
from .codegen.cpp.generator import CppCodeGenerator
from .codegen.rust.generator import RustCodeGenerator

__all__ = ['ModelParser', 'CppCodeGenerator', 'RustCodeGenerator']
