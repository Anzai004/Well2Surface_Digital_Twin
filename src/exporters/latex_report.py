"""
Automated LaTeX / PDF cycle-summary report generator.

Renders a simulation state payload into a self-contained LaTeX document and
attempts to compile it to PDF via a local `pdflatex` (MiKTeX / TeX Live).
Compilation is best-effort: if no LaTeX toolchain is present on the host,
the .tex source is still written and returned so the pipeline never hard
fails on a missing binary.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _escape_latex(value: Any) -> str:
    text = str(value)
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return _escape_latex(value)


_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{siunitx}
\title{Well2Surface Digital Twin --- Cycle Summary}
\author{Baghewala Field Autonomous Control Engine}
\date{__TIMESTAMP__}
\begin{document}
\maketitle

\section*{Well: __WELL_ID__}

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
__ROWS__
\bottomrule
\end{tabular}
\caption{Cycle-end simulation state snapshot.}
\end{table}

\end{document}
"""


def generate_latex_report(
    state: Dict[str, Any],
    output_path: Union[str, Path],
    compile_pdf: bool = True,
) -> Optional[Path]:
    """
    Renders `state` into a LaTeX summary and writes it alongside
    `output_path` (with a .tex extension). If a `pdflatex` binary is
    available on PATH, also compiles it to PDF at `output_path`.

    Args:
        state: Flat dict of cycle-summary key/value pairs.
        output_path: Desired output path -- may end in .pdf or .tex; the
            .tex source is always written next to it.
        compile_pdf: If True (default), attempt a pdflatex compile when the
            binary is available. Silently skipped otherwise.

    Returns:
        Path to the compiled PDF if compilation succeeded, else None (the
        .tex source is always written regardless).
    """
    output_path = Path(output_path)
    tex_path = output_path.with_suffix(".tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    rows = "\n".join(
        f"{_escape_latex(key)} & {_fmt(value)} \\\\"
        for key, value in state.items()
    )

    tex_source = (
        _TEMPLATE
        .replace("__TIMESTAMP__", _escape_latex(state.get("timestamp_utc", "")))
        .replace("__WELL_ID__", _escape_latex(state.get("well_id", "Unknown")))
        .replace("__ROWS__", rows)
    )

    tex_path.write_text(tex_source, encoding="utf-8")

    if not compile_pdf:
        return None

    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin is None:
        return None

    try:
        subprocess.run(
            [pdflatex_bin, "-interaction=nonstopmode", "-output-directory",
             str(tex_path.parent), str(tex_path)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    compiled_pdf = tex_path.with_suffix(".pdf")
    return compiled_pdf if compiled_pdf.exists() else None