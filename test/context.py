import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simplex import simplex
from src.revised_simplex import revised_simplex
from src.Model import Model