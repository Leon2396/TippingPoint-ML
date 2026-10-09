"""
Root CLI entrypoint for TippingPoint ML Engine testing.
Delegates directly to ml_engine.test_ml.
"""

import sys
import os

# Self-healing environment check for Windows Application Control on Python 3.14
if os.environ.get("_TIPPINGPOINT_RELAUNCHED") is None:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        import subprocess
        candidates = [
            os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"),
            "py -3.13",
        ]
        relaunch_cmd = None
        for cand in candidates:
            if cand.startswith("py "):
                relaunch_cmd = ["py", "-3.13", __file__] + sys.argv[1:]
                break
            elif os.path.isfile(cand):
                relaunch_cmd = [cand, __file__] + sys.argv[1:]
                break
        if relaunch_cmd:
            env = os.environ.copy()
            env["_TIPPINGPOINT_RELAUNCHED"] = "1"
            res = subprocess.run(relaunch_cmd, env=env)
            sys.exit(res.returncode)

from ml_engine.test_ml import main

if __name__ == "__main__":
    main()
