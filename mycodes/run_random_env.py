import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# bin_packing_env.py owns the IsaacLab AppLauncher.
# Importing it launches the simulation app and parses environment arguments.
import bin_packing_env as bpe  # type: ignore  # noqa: E402


def main():
    """Run random-policy validation for the bin-packing RL environment.

    This script is used to validate:
    - reset()
    - step(action)
    - reward computation
    - done/success logic
    - vector/image/multimodal observation modes
    - optional camera output

    PPO training is handled separately by train.py.
    """
    bpe.main()


if __name__ == "__main__":
    main()