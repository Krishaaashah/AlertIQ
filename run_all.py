"""
AlertIQ — Full Pipeline Orchestrator
Runs all phases sequentially: Data → Model → RL → Drift → Done.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    start = time.time()

    print("\n" + "█" * 60)
    print("  AlertIQ — Full Pipeline Execution")
    print("█" * 60)

    # Phase 2
    print("\n\n" + "▶" * 30 + " PHASE 2")
    from run_phase2 import main as run_p2
    run_p2()

    # Phase 3
    print("\n\n" + "▶" * 30 + " PHASE 3")
    from run_phase3 import main as run_p3
    run_p3()

    # Phase 4
    print("\n\n" + "▶" * 30 + " PHASE 4")
    from run_phase4_rl import main as run_p4
    run_p4()

    # Phase 5
    print("\n\n" + "▶" * 30 + " PHASE 5")
    from run_phase5_drift import main as run_p5
    run_p5()

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n\n" + "█" * 60)
    print(f"  AlertIQ — All Phases Complete ({minutes}m {seconds}s)")
    print("█" * 60)
    print(f"\n  Outputs:  outputs/")
    print(f"  Models:   models/")
    print(f"  RL:       outputs/rl_outputs/")
    print(f"\n  To start the API:  python -m uvicorn api.main:app --reload")
    print("█" * 60)


if __name__ == "__main__":
    main()
