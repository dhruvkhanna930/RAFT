"""Pytest configuration for the unicode-rag-poison test suite.

Sets KMP_DUPLICATE_LIB_OK=TRUE *before* any C-extension is imported so
that the dual-OpenMP conflict between PyTorch and FAISS (both ship their
own libomp.dylib on macOS) does not abort the process during search().
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
