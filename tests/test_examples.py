"""Smoke test for all example scripts in examples/ directory."""
import os
import sys
import pytest
import glob
import subprocess

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

# Scripts that require large models or significant GPU/memory — skip in CI
HEAVY_EXAMPLES = {
    "hessian_cnn_loss_landscape.py",  # downloads VGG16 (528MB) and computes Hessian on 138M params
}

example_scripts = glob.glob(os.path.join(EXAMPLES_DIR, "*.py"))

@pytest.mark.parametrize("script_path", example_scripts)
def test_example_script_runs(script_path):
    """Smoke test: run each example script and check for errors."""
    if os.path.basename(script_path) in HEAVY_EXAMPLES:
        pytest.skip(f"Skipping {os.path.basename(script_path)}: requires large model/memory, not suitable for CI")

    print(f"Running example: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

    # Check if failure is due to missing optional dependencies
    if result.returncode != 0:
        stderr = result.stderr.lower()
        # Skip tests that fail due to missing optional dependencies
        if any(missing_dep in stderr for missing_dep in [
            'no module named \'torchvision\'',
            'no module named \'transformers\'',
            'no module named \'datasets\'',
            'no module named \'streamlit\'',
            'no module named \'plotly\''
        ]):
            pytest.skip(f"Skipping {script_path} due to missing optional dependency")

    assert result.returncode == 0, f"Script {script_path} failed with error:\n{result.stderr}"

