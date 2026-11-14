from pathlib import Path
import subprocess
import sys


# Paths to the Sahuaro repository and the IRMOF-1 example.
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "Examples" / "Flexible_IRMOF-1_Dubbeldam2007"


def read_lines(path, skip_header=False):
    lines = path.read_text().splitlines()

    # Skip the first line of in.data, which contains the generation date.
    if skip_header:
        lines = lines[1:]

    # Ignore differences in spacing between the generated and reference files.
    return [" ".join(line.split()) for line in lines]


def test_irmof1_conversion(tmp_path):
    config = EXAMPLE / "0_input" / "sahuaro_input"

    # Run Sahuaro using the example input and write the files to a temporary folder.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sahuaro" / "Sahuaro.py"),
            str(config),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    # Check that the conversion finished without errors.
    assert result.returncode == 0, result.stdout + result.stderr

    # Compare the generated LAMMPS files with the saved example outputs.
    for filename in ("in.data", "in.coeffs"):
        generated = tmp_path / filename
        reference = EXAMPLE / "1_output" / filename

        assert generated.is_file()

        skip_header = filename == "in.data"

        assert read_lines(
            generated, skip_header
        ) == read_lines(
            reference, skip_header
        )
