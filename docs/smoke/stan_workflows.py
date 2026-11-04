"""Run the published Stan examples with short trajectories in either language."""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


DOCS = Path(__file__).resolve().parents[1]
PAGE = DOCS / "src/examples/stan-workflows.md"
FIXTURES = DOCS / "src/public/stan"
EXAMPLES = (
    ("## A multivariate normal: Zig-Zag and two event clocks", "## Independent spike-and-slab prior", "normal"),
    ("## Independent spike-and-slab prior", "### Beta-binomial inclusion with independent slabs", "independent"),
    ("### Beta-binomial inclusion with independent slabs", "## Correlated slab and beta-binomial model prior", "beta_binomial"),
    ("## Logistic regression with variable selection", "## A Bayesian two-sample test", "logistic"),
    ("## A Bayesian two-sample test", "## Saving the result", "ttest"),
)


def example_code(page, heading, next_heading, language):
    section = page.split(heading, 1)[1].split(next_heading, 1)[0]
    fence = "r" if language == "r" else "julia"
    label = "R" if language == "r" else "Julia"
    code = re.search(r"== " + label + r"\n\n```" + fence + r"\n(.*?)\n```", section, re.S)
    if code is None:
        raise ValueError(f"Missing {label} example: {heading}")
    result = code.group(1)
    if language == "r":
        result = re.sub(r"\bT = (?:10000|20000|100000)\b", "T = 4", result)
    else:
        for length in ("10_000.0", "20_000.0", "100_000.0"):
            result = result.replace(length, "4.0")
    return result + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("r", "julia"), required=True)
    args = parser.parse_args()
    page = PAGE.read_text()
    with tempfile.TemporaryDirectory(prefix="pdmp-stan-smoke-") as tmp:
        work = Path(tmp)
        for name in ("mvnormal", "logistic_regression", "ttest"):
            if name == "mvnormal":
                shutil.copyfile(FIXTURES / "mvnormal.stan", work / "mvnormal.stan")
                continue
            for suffix in (".stan", "_data.json"):
                shutil.copyfile(FIXTURES / (name + suffix), work / (name + suffix))
        blocks = []
        for heading, next_heading, name in EXAMPLES:
            code = example_code(page, heading, next_heading, args.language)
            if args.language == "r":
                blocks.append(f'cat("Running R {name}\\n")\nlocal({{\n{code}}})\n')
            else:
                blocks.append(f'println("Running Julia {name}")\nmodule Smoke_{name}\n{code}end\n')
        script = work / ("stan_workflows.R" if args.language == "r" else "stan_workflows.jl")
        script.write_text("\n".join(blocks))
        command = (["Rscript", str(script)] if args.language == "r" else
                   ["julia", f"--project={DOCS}", str(script)])
        subprocess.run(command, cwd=work, check=True)


if __name__ == "__main__":
    main()
