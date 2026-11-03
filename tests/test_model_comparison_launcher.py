import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("model_type,expected_mode", [("qwen2", None), ("qwen3_5_moe", "align")])
def test_launcher_enables_mamba_alignment_only_for_hybrid_model(tmp_path, model_type, expected_mode):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    captured = tmp_path / "arguments.json"
    shim = tmp_path / "python"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'vllm.entrypoints.cli.main']:\n"
        "    with open(os.environ['CAPTURE_ARGUMENTS'], 'w') as f: json.dump(sys.argv[3:], f)\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    shim.chmod(0o755)
    environment = dict(os.environ, MODEL_PATH=str(model), CAPTURE_ARGUMENTS=str(captured))
    environment["PATH"] = str(tmp_path) + os.pathsep + environment["PATH"]
    subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "scripts/model-comparison/serve.sh")],
        env=environment,
        check=True,
    )
    args = json.loads(captured.read_text())
    actual = args[args.index("--mamba-cache-mode") + 1] if "--mamba-cache-mode" in args else None
    assert actual == expected_mode
