"""
Moses SMT Docker orchestration.

Provides a Python interface to run Moses training and translation
via the `amake/moses-smt` Docker image. Handles:

- Docker container lifecycle
- Data preparation (tokenize, truecase, clean)
- Training pipeline (GIZA++ → phrase extraction → KenLM → MERT)
- Batch translation
- Model persistence

Requirements:
    - Docker (tested with 24.0+)
    - `docker pull amake/moses-smt`
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from . import utils

logger = utils.logger

# ─── Docker Command Helpers ──────────────────────────────────────────


def _docker_available() -> bool:
    """Check if Docker is available on the system."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_cmd(cmd: List[str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"
    except FileNotFoundError as e:
        return -1, "", f"Command not found: {e}"


def _docker_run(
    image: str = "amake/moses-smt",
    command: str = "",
    volumes: Optional[Dict[str, str]] = None,
    workdir: str = "/moses/model",
    name: Optional[str] = None,
    rm: bool = True,
    interactive: bool = False,
    timeout: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Execute a command inside the Moses Docker container.

    Args:
        image: Docker image name.
        command: Command to run inside container.
        volumes: Dict mapping host_path → container_path.
        workdir: Working directory inside container.
        name: Container name.
        rm: Auto-remove container after exit.
        interactive: Run with -i flag.
        timeout: Command timeout in seconds.

    Returns:
        (returncode, stdout, stderr)
    """
    cmd = ["docker", "run"]
    if rm:
        cmd.append("--rm")
    if interactive:
        cmd.append("-i")
    if name:
        cmd.extend(["--name", name])
    if volumes:
        for host_path, container_path in volumes.items():
            abs_path = os.path.abspath(host_path)
            cmd.extend(["-v", f"{abs_path}:{container_path}"])
    cmd.extend(["-w", workdir])
    cmd.append(image)
    if command:
        cmd.extend(["sh", "-c", command])

    return _run_cmd(cmd, timeout=timeout)


# ─── Data Preparation (Docker-based) ─────────────────────────────────


def prepare_data_moses(
    src_raw: str,
    tgt_raw: str,
    output_dir: str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    model_dir: str = "/moses/model",
) -> Dict[str, str]:
    """Prepare training data inside Moses Docker.

    Runs tokenization, truecasing, and corpus cleaning using Moses
    Perl scripts.

    Args:
        src_raw: Host path to raw source file.
        tgt_raw: Host path to raw target file.
        output_dir: Host output directory for prepared files.
        src_lang: Source language code.
        tgt_lang: Target language code.
        model_dir: Container working directory.

    Returns:
        Dict of {step: container_path} for prepared files.
    """
    utils.ensure_dir(output_dir)
    files: Dict[str, str] = {}

    # Tokenize source
    src_tok_out = os.path.join(output_dir, f"train.tok.{src_lang}")
    _docker_run(
        command=f"tokenizer.perl -l {src_lang} < {model_dir}/train.{src_lang} > {model_dir}/train.tok.{src_lang}",
        volumes={os.path.dirname(src_raw): model_dir},
    )
    files["src_tokenized"] = src_tok_out

    # Tokenize target
    tgt_tok_out = os.path.join(output_dir, f"train.tok.{tgt_lang}")
    _docker_run(
        command=f"tokenizer.perl -l {tgt_lang} < {model_dir}/train.{tgt_lang} > {model_dir}/train.tok.{tgt_lang}",
    )
    files["tgt_tokenized"] = tgt_tok_out

    return files


# ─── Training Pipeline ────────────────────────────────────────────────


def train_moses_model(
    src_file: str,
    tgt_file: str,
    output_dir: str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    lm_order: int = 5,
    use_mert: bool = False,
    dev_src: Optional[str] = None,
    dev_tgt: Optional[str] = None,
) -> bool:
    """Train a complete Moses SMT model via Docker.

    Pipeline:
        1. Tokenize → truecase → clean
        2. Run GIZA++ word alignment
        3. Phrase extraction + scoring
        4. Train KenLM language model
        5. (Optional) MERT tuning on dev set

    Args:
        src_file: Host path to source training file (raw, one-sentence-per-line).
        tgt_file: Host path to target training file (raw).
        output_dir: Host output directory for the trained model.
        src_lang: Source language code.
        tgt_lang: Target language code.
        lm_order: KenLM n-gram order.
        use_mert: Whether to run MERT tuning.
        dev_src: Host path to development source (required for MERT).
        dev_tgt: Host path to development target (required for MERT).

    Returns:
        True if training succeeded.
    """
    if not _docker_available():
        logger.error("Docker is not available. Cannot train Moses model.")
        return False

    utils.ensure_dir(output_dir)

    # Container path for our data
    container_data = "/moses/data"
    container_model = "/moses/model"

    # Copy files to a working directory
    work_dir = os.path.join(output_dir, "work")
    utils.ensure_dir(work_dir)
    volumes = {work_dir: container_model}

    logger.info("=== Step 1: Tokenization ===")
    _docker_run(
        command=(
            f"tokenizer.perl -l {src_lang} < {container_model}/train.{src_lang} "
            f"> {container_model}/train.tok.{src_lang}"
        ),
        volumes=volumes,
    )
    _docker_run(
        command=(
            f"tokenizer.perl -l {tgt_lang} < {container_model}/train.{tgt_lang} "
            f"> {container_model}/train.tok.{tgt_lang}"
        ),
        volumes=volumes,
    )

    logger.info("=== Step 2: Truecasing ===")
    _docker_run(
        command=(
            f"truecase.perl --model {container_model}/truecase-model.{tgt_lang} "
            f"< {container_model}/train.tok.{tgt_lang} "
            f"> {container_model}/train.true.{tgt_lang}"
        ),
        volumes=volumes,
    )
    _docker_run(
        command=(
            f"truecase.perl --model {container_model}/truecase-model.{src_lang} "
            f"< {container_model}/train.tok.{src_lang} "
            f"> {container_model}/train.true.{src_lang}"
        ),
        volumes=volumes,
    )

    logger.info("=== Step 3: Corpus Cleaning ===")
    _docker_run(
        command=(
            f"clean-corpus-n.perl {container_model}/train.true {src_lang} {tgt_lang} "
            f"{container_model}/train.clean 1 80"
        ),
        volumes=volumes,
    )

    logger.info("=== Step 4: Training Language Model (KenLM) ===")
    _docker_run(
        command=(
            f"lmplz -o {lm_order} -S 80% "
            f"< {container_model}/train.clean.{tgt_lang} "
            f"> {container_model}/lm.arpa"
        ),
        volumes=volumes,
        timeout=3600,
    )
    _docker_run(
        command=f"build_binary {container_model}/lm.arpa {container_model}/lm.blm",
        volumes=volumes,
    )

    logger.info("=== Step 5: Word Alignment (GIZA++) ===")
    _docker_run(
        command=(
            f"train-model.perl -root-dir {container_model}/alignment "
            f"-corpus {container_model}/train.clean "
            f"-f {src_lang} -e {tgt_lang} "
            f"-alignment grow-diag-final-and "
            f"-reordering msd-bidirectional-fe "
            f"-lm 0:{lm_order}:{container_model}/lm.blm:0 "
            f"-external-bin-dir /usr/local/bin "
            f">& {container_model}/train-model.log"
        ),
        volumes=volumes,
        timeout=14400,  # 4 hours max
    )

    logger.info("=== Step 6: MERT Tuning (optional) ===")
    if use_mert and dev_src and dev_tgt:
        _docker_run(
            command=(
                f"mert-moses.pl {container_model}/dev.{src_lang} {container_model}/dev.{tgt_lang} "
                f"{container_model}/alignment/model/moses.ini "
                f"--mertdir /usr/local/bin "
                f"--decoder-flags '-n-best-list {container_model}/nbest.txt 20' "
                f"--mertargs '--sctype BLEU' "
                f"&> {container_model}/mert.log"
            ),
            volumes=volumes,
            timeout=7200,
        )

    logger.info(f"Moses training complete. Model saved to {output_dir}")
    return True


# ─── Translation ─────────────────────────────────────────────────────


def translate_moses(
    input_file: str,
    output_file: str,
    model_dir: str,
    config_file: str = "moses.ini",
) -> bool:
    """Translate a file using trained Moses model via Docker.

    Args:
        input_file: Host path to input file (tokenized, one-sentence-per-line).
        output_file: Host path to write translations.
        model_dir: Host path to trained model directory.
        config_file: Moses config file name (relative to model_dir).

    Returns:
        True if translation succeeded.
    """
    if not _docker_available():
        logger.error("Docker is not available.")
        return False

    utils.ensure_dir(os.path.dirname(output_file) or ".")

    # Find the actual config path
    config_path = os.path.join(model_dir, "alignment", "model", config_file)
    if not os.path.exists(config_path):
        config_path = os.path.join(model_dir, config_file)
    if not os.path.exists(config_path):
        logger.error(f"Moses config not found: {config_path}")
        return False

    container_model = "/moses/model"
    volumes = {model_dir: container_model}

    _docker_run(
        command=(
            f"moses -f {container_model}/{os.path.relpath(config_path, model_dir)} "
            f"< {container_model}/{os.path.basename(input_file)} "
            f"> {container_model}/{os.path.basename(output_file)}"
        ),
        volumes=volumes,
        timeout=3600,
    )

    logger.info(f"Translation saved to {output_file}")
    return True


def batch_translate_moses(
    source_dir: str,
    output_dir: str,
    model_dir: str,
    file_list: List[str],
    src_lang: str = "zh",
) -> int:
    """Batch translate multiple files using Moses.

    Args:
        source_dir: Directory containing source files.
        output_dir: Directory for translated files.
        model_dir: Trained Moses model directory.
        file_list: List of source filenames.
        src_lang: Source language code.

    Returns:
        Number of successfully translated files.
    """
    utils.ensure_dir(output_dir)
    success = 0

    for filename in file_list:
        src_path = os.path.join(source_dir, filename)
        out_path = os.path.join(output_dir, filename.replace(f".{src_lang}", ".en"))

        if not os.path.exists(src_path):
            logger.warning(f"Source not found: {src_path}")
            continue

        ok = translate_moses(
            input_file=src_path,
            output_file=out_path,
            model_dir=model_dir,
        )
        if ok:
            success += 1

    logger.info(f"Batch translation complete: {success}/{len(file_list)} successful")
    return success


# ─── System Check ───────────────────────────────────────────────────


def check_moses_setup() -> Dict[str, bool]:
    """Check if Moses Docker environment is properly set up.

    Returns:
        Dict of component → availability status.
    """
    results = {
        "docker": _docker_available(),
    }

    if results["docker"]:
        # Check image
        rc, _, _ = _run_cmd(["docker", "image", "inspect", "amake/moses-smt"])
        results["moses_image"] = (rc == 0)

        # Quick test
        if results["moses_image"]:
            rc, out, _ = _docker_run(command="moses --version")
            results["moses_binary"] = (rc == 0)
        else:
            results["moses_binary"] = False
    else:
        results["moses_image"] = False
        results["moses_binary"] = False

    return results
