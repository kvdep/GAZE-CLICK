import sys, os
_mvp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if _mvp_dir not in sys.path:
    sys.path.insert(0, _mvp_dir)
