"""Collect repeated LLM ratings for every company-sub-criterion unit.

Two operational constraints shape this module, both measured on the target
machine rather than assumed:

* At ``temperature=0`` the seed has no effect and every iteration returns
  byte-identical output, which would make all per-iteration agreement values the
  same number. Iteration variance therefore requires ``temperature > 0``, with
  ``seed = iteration`` keeping each iteration individually reproducible.
* Swapping models costs a 280-966 s reload, against 12-26 s for a warm call. The
  model loop is therefore the outermost loop and ``keep_alive`` is held open, so
  each model is loaded exactly once.

Results append to JSONL and the run resumes from whatever is already on disk.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence
from urllib import error, request
from urllib.parse import urlsplit

from src.decision_support_mcp.rating_evaluation.dataset import Unit
from src.decision_support_mcp.rating_evaluation.prompts import (
    MAX_RATING,
    MIN_RATING,
    PROMPT_SHA256,
    RUBRIC_SHA256,
    SYSTEM_PROMPT,
    build_prompt,
)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 1.0
DEFAULT_KEEP_ALIVE = "60m"
DEFAULT_MAX_ATTEMPTS = 3
#: Wall-clock ceiling per call, enforced by :func:`_post_json`. Measured on this
#: machine, the slowest legitimate answer was gemma4 at 90 s and the stall mode
#: starts at 290 s, so this sits in the empty gap between them. It is a real
#: deadline, not the socket timeout it replaced.
DEFAULT_TIMEOUT_SEC = 120.0


class ModelSpec(NamedTuple):
    """One model in the comparison. ``tag`` must be an exact Ollama tag."""

    family: str
    tag: str

    @classmethod
    def parse(cls, text: str) -> "ModelSpec":
        """Parse ``Family=tag`` or a bare ``tag``."""

        if "=" in text:
            family, tag = text.split("=", 1)
            return cls(family.strip(), tag.strip())
        tag = text.strip()
        return cls(tag.split(":", 1)[0].capitalize(), tag)


def list_installed_models(base_url: str = DEFAULT_BASE_URL) -> dict[str, str]:
    """Map installed Ollama tag -> digest. Raises if Ollama is unreachable."""

    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with request.urlopen(request.Request(url, method="GET"), timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. Start it with `ollama serve`."
        ) from exc
    return {
        str(item.get("name")): str(item.get("digest", ""))[:12]
        for item in payload.get("models", [])
    }


def verify_models(
    models: Sequence[ModelSpec],
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Check every requested tag is installed, without pulling anything."""

    installed = list_installed_models(base_url)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for spec in models:
        if spec.tag in installed:
            resolved[spec.tag] = installed[spec.tag]
        else:
            missing.append(spec.tag)
    return {
        "installed": installed,
        "resolved": resolved,
        "missing": missing,
        "ok": not missing,
    }


def unload_model(tag: str, base_url: str = DEFAULT_BASE_URL) -> bool:
    """Evict a model from memory, the API equivalent of ``ollama stop <tag>``.

    Sending ``keep_alive: 0`` with an empty prompt frees the weights immediately
    so the next model does not have to compete for memory. Returns False if the
    request failed, which is not fatal: the model simply stays resident.
    """

    body = {"model": tag, "prompt": "", "stream": False, "keep_alive": 0}
    req = request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            response.read()
        return True
    except (error.HTTPError, error.URLError, TimeoutError):
        return False


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise ValueError("Response contains no JSON object.")
        stripped = match.group(0)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Response JSON is not an object.")
    return parsed


def _parse_rating(payload: dict[str, Any]) -> tuple[int, str]:
    rating = payload.get("rating")
    justification = payload.get("justification") or payload.get("rationale")
    if isinstance(rating, str):
        rating = rating.strip().replace(",", ".")
    try:
        numeric = float(rating)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rating {rating!r} is not numeric.") from exc
    if numeric != int(numeric):
        raise ValueError(f"rating {numeric} is not an integer.")
    integer = int(numeric)
    if not MIN_RATING <= integer <= MAX_RATING:
        raise ValueError(f"rating {integer} is outside {MIN_RATING}-{MAX_RATING}.")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError("justification is missing or empty.")
    return integer, justification.strip()


class DeadlineExceeded(RuntimeError):
    """A call outlived its wall-clock budget and was cut off at the socket."""


def _post_json(url: str, body: dict[str, Any], deadline_sec: float) -> dict[str, Any]:
    """POST and decode JSON under a hard wall-clock deadline.

    ``urlopen``'s ``timeout`` is a *socket* timeout: it bounds each individual
    read, not the request as a whole. A generation that stalls after the server
    has answered, or that trickles, therefore runs unbounded -- measured on this
    study, calls of 946 s under a nominal 120 s timeout, none of them retried
    because none of them ever raised.

    A watchdog thread supplies the ceiling the socket timeout only appears to
    give. It shuts the socket down rather than merely closing it, because
    ``close()`` does not wake a thread already blocked in ``recv``; the shutdown
    also reaches Ollama as a client disconnect, which cancels the generation
    instead of leaving it holding the GPU while the retry queues behind it.
    """

    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    # The socket timeout is only a backstop for a connect that never lands. It
    # is deliberately slack against the deadline so the watchdog always fires
    # first: the watchdog's shutdown is what tells Ollama to cancel, whereas a
    # socket timeout leaves the generation running.
    conn = HTTPConnection(
        parts.hostname or "localhost", parts.port or 80, timeout=deadline_sec * 2 + 30
    )
    expired = threading.Event()

    def abort() -> None:
        expired.set()
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                # Already torn down by the request thread; nothing to wake.
                pass

    watchdog = threading.Timer(deadline_sec, abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        conn.request(
            "POST", path,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        status = response.status
    except (OSError, HTTPException) as exc:
        # A socket timeout means the same thing to the caller as the watchdog
        # firing, so both surface as one exception type.
        if expired.is_set() or isinstance(exc, TimeoutError):
            raise DeadlineExceeded(
                f"no complete response within {deadline_sec:.0f}s; the socket was "
                "closed and the generation cancelled"
            ) from exc
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc
    finally:
        # Cancel before closing: a watchdog that fires here would shut down a
        # socket whose response is already in hand.
        watchdog.cancel()
        conn.close()

    if status != 200:
        detail = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {status} from Ollama: {detail[:300]}")
    return json.loads(raw.decode("utf-8"))


def _nanos_to_sec(value: Any) -> float | None:
    """Ollama reports every duration in nanoseconds."""

    if value is None:
        return None
    try:
        return round(int(value) / 1e9, 4)
    except (TypeError, ValueError):
        return None


def _call_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """The token counters and timings Ollama returns alongside the answer.

    ``prompt_eval_count`` is absent when the prompt was served from the prefix
    cache, so a missing value means "not re-evaluated", not zero.
    """

    prompt_tokens = payload.get("prompt_eval_count")
    completion_tokens = payload.get("eval_count")
    eval_sec = _nanos_to_sec(payload.get("eval_duration"))
    prompt_sec = _nanos_to_sec(payload.get("prompt_eval_duration"))
    load_sec = _nanos_to_sec(payload.get("load_duration"))

    total_tokens: int | None = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # Throughput of the generation phase alone, the figure that is
        # comparable across models of different sizes.
        "tokens_per_sec": (
            round(completion_tokens / eval_sec, 2)
            if completion_tokens and eval_sec
            else None
        ),
        "load_duration_sec": load_sec,
        "prompt_eval_duration_sec": prompt_sec,
        "eval_duration_sec": eval_sec,
        "total_duration_sec": _nanos_to_sec(payload.get("total_duration")),
        # "length" means the answer hit the token ceiling and was cut off, which
        # is a truncated rating rather than a refused one.
        "done_reason": payload.get("done_reason"),
    }


def _generate(
    base_url: str,
    tag: str,
    prompt: str,
    temperature: float,
    top_p: float,
    seed: int,
    keep_alive: str,
    timeout_sec: float,
    disable_thinking: bool = True,
) -> tuple[str, float, dict[str, Any]]:
    """One scoring call. Returns the answer, the wall-clock latency and metrics.

    ``timeout_sec`` is a hard deadline: past it the socket is shut down and
    :class:`DeadlineExceeded` is raised, which the caller's retry loop treats
    like any other failed attempt.

    ``disable_thinking`` sends ``think: false``. Reasoning models route their
    whole output into a separate ``thinking`` field and leave ``response`` empty,
    which reads downstream as a total parse failure rather than as a
    configuration problem. Measured on the study models: qwen3.5:9b returns
    nothing at all without this flag and gemma4:12b returns unparseable text,
    while llama3.1:8b and deepseek-v2:16b return byte-identical output with or
    without it. Sending it uniformly therefore keeps one request shape across all
    four models without changing what the non-reasoning ones produce.
    """

    body: dict[str, Any] = {
        "model": tag,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "top_p": top_p, "seed": seed},
    }
    if disable_thinking:
        body["think"] = False
    if SYSTEM_PROMPT:
        body["system"] = SYSTEM_PROMPT
    started = time.time()
    payload = _post_json(f"{base_url.rstrip('/')}/api/generate", body, timeout_sec)

    text = str(payload.get("response", "") or "")
    if not text.strip():
        # Defensive: a model that ignores think:false still leaves its output in
        # the thinking field, which may hold the JSON object we asked for.
        thinking = str(payload.get("thinking", "") or "")
        if thinking.strip():
            text = thinking
    return text, time.time() - started, _call_metrics(payload)


def check_prompt_provenance(path: Path) -> dict[str, Any]:
    """Compare the hashes already on disk with the prompt about to be used.

    Ratings collected under different rubrics are not comparable, and because
    collection appends and resumes, a silently edited rubric would interleave two
    incompatible scales in one file. Every record stores the hashes precisely so
    that mismatch is detectable instead of invisible.
    """

    found_rubric: set[str] = set()
    found_prompt: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("rubric_sha256"):
                    found_rubric.add(str(record["rubric_sha256"]))
                if record.get("prompt_sha256"):
                    found_prompt.add(str(record["prompt_sha256"]))

    stale_rubric = found_rubric - {RUBRIC_SHA256}
    stale_prompt = found_prompt - {PROMPT_SHA256}
    return {
        "ok": not stale_rubric and not stale_prompt,
        "existing_records": bool(found_rubric or found_prompt),
        "stale_rubric_hashes": sorted(stale_rubric),
        "stale_prompt_hashes": sorted(stale_prompt),
        "current_rubric_sha256": RUBRIC_SHA256,
        "current_prompt_sha256": PROMPT_SHA256,
    }


def load_completed(path: Path) -> set[tuple[str, int, str, str]]:
    """Keys already on disk, so a resumed run skips them."""

    done: set[tuple[str, int, str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A partial trailing line from an interrupted run. The unit is
                # simply re-scored.
                continue
            if record.get("rating") is None:
                continue
            done.add(
                (
                    str(record["model_tag"]),
                    int(record["iteration"]),
                    str(record["company"]),
                    str(record["code"]),
                )
            )
    return done


def load_ratings(path: Path) -> list[dict[str, Any]]:
    """Every stored rating record, successful or not."""

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def plan(
    units: Sequence[Unit],
    models: Sequence[ModelSpec],
    iterations: int,
    completed: set[tuple[str, int, str, str]],
) -> Iterator[tuple[ModelSpec, int, Unit]]:
    """Yield outstanding work, model-major so each model loads once."""

    for spec in models:
        for iteration in range(1, iterations + 1):
            for unit in units:
                if (spec.tag, iteration, unit.company, unit.code) in completed:
                    continue
                yield spec, iteration, unit


def collect(
    units: Sequence[Unit],
    models: Sequence[ModelSpec],
    iterations: int,
    out_path: Path,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    progress_every: int = 10,
    verify: bool = True,
    unload_between_models: bool = True,
    disable_thinking: bool = True,
    progress_callback: Any = None,
    should_stop: Any = None,
) -> dict[str, Any]:
    """Run outstanding scoring calls and append them to ``out_path``."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    if temperature <= 0.0:
        # Deliberate configuration for a determinism check. Greedy decoding
        # ignores the seed, so every iteration reproduces iteration 1 and the
        # per-iteration agreement statistics collapse to a single repeated value.
        print(
            f"NOTE temperature={temperature}: Ollama decodes greedily and ignores the "
            f"seed, so all {iterations} iterations are expected to be identical. "
            "Per-iteration alpha and kappa will have zero spread by construction.",
            file=sys.stderr,
            flush=True,
        )

    digests: dict[str, str] = {}
    if verify:
        check = verify_models(models, base_url)
        if not check["ok"]:
            raise RuntimeError(
                "These model tags are not installed: "
                + ", ".join(check["missing"])
                + ". Installed: "
                + (", ".join(sorted(check["installed"])) or "none")
                + ". Pull them with `ollama pull <tag>` before collecting."
            )
        digests = check["resolved"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = check_prompt_provenance(out_path)
    if not provenance["ok"]:
        raise RuntimeError(
            f"{out_path} holds ratings collected under a different prompt or rubric "
            f"(rubric {provenance['stale_rubric_hashes']}, "
            f"prompt {provenance['stale_prompt_hashes']}); the current pair is "
            f"rubric {RUBRIC_SHA256[:12]}, prompt {PROMPT_SHA256[:12]}. Scores from "
            "two different rubrics are not comparable and resuming would interleave "
            "them. Move the existing file aside and start a fresh collection."
        )

    completed = load_completed(out_path)
    outstanding = list(plan(units, models, iterations, completed))
    total = len(outstanding)

    counters = {
        "written": 0, "failed": 0, "retried": 0, "truncated": 0,
        "deadline_aborts": 0, "prompt_tokens": 0, "completion_tokens": 0,
    }
    started_at = time.time()
    unloaded: list[str] = []
    active_tag: str | None = None

    stopped = False
    with out_path.open("a", encoding="utf-8") as handle:
        for index, (spec, iteration, unit) in enumerate(outstanding, start=1):
            if should_stop is not None and should_stop():
                # Cancellation lands between calls, never mid-write, so the JSONL
                # stays valid and the run resumes from exactly here.
                stopped = True
                break
            if unload_between_models and active_tag is not None and spec.tag != active_tag:
                # The work list is model-major, so a tag change means the previous
                # model is finished and its weights can be freed.
                if unload_model(active_tag, base_url):
                    unloaded.append(active_tag)
                print(f"unloaded {active_tag}", file=sys.stderr, flush=True)
            active_tag = spec.tag

            prompt = build_prompt(unit.measure, unit.guidance, unit.disclosure)
            rating: int | None = None
            justification: str | None = None
            raw = ""
            latency = 0.0
            metrics: dict[str, Any] = {}
            failure: str | None = None
            attempts = 0

            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                # Retries perturb the seed: repeating an identical deterministic
                # request would reproduce the same unparseable output.
                seed = iteration + 100_000 * (attempt - 1)
                try:
                    raw, latency, metrics = _generate(
                        base_url, spec.tag, prompt, temperature, top_p,
                        seed, keep_alive, timeout_sec, disable_thinking,
                    )
                    rating, justification = _parse_rating(_extract_json_object(raw))
                    failure = None
                    break
                except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, DeadlineExceeded):
                        counters["deadline_aborts"] += 1
                    if attempt < max_attempts:
                        counters["retried"] += 1
                        time.sleep(min(2.0 * attempt, 10.0))

            record = {
                "family": spec.family,
                "model_tag": spec.tag,
                "model_digest": digests.get(spec.tag),
                "iteration": iteration,
                "seed": iteration,
                "temperature": temperature,
                "top_p": top_p,
                "format": "json",
                "think": (False if disable_thinking else None),
                "company": unit.company,
                "code": unit.code,
                "rating": rating,
                "justification": justification,
                "attempts": attempts,
                "failure": failure,
                "latency_sec": round(latency, 3),
                # Token counts, per-phase timings and done_reason, straight from
                # the last attempt's Ollama response.
                **metrics,
                "justification_chars": len(justification or ""),
                "raw_response": raw,
                "prompt_sha256": PROMPT_SHA256,
                "rubric_sha256": RUBRIC_SHA256,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            if rating is None:
                counters["failed"] += 1
            else:
                counters["written"] += 1
            counters["prompt_tokens"] += int(metrics.get("prompt_tokens") or 0)
            counters["completion_tokens"] += int(metrics.get("completion_tokens") or 0)
            if metrics.get("done_reason") == "length":
                counters["truncated"] += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "done": index,
                        "total": total,
                        "model_tag": spec.tag,
                        "family": spec.family,
                        "iteration": iteration,
                        "company": unit.company,
                        "code": unit.code,
                        "rating": rating,
                        "failure": failure,
                        "latency_sec": round(latency, 3),
                        "total_tokens": metrics.get("total_tokens"),
                        "elapsed_sec": round(time.time() - started_at, 1),
                    }
                )

            if progress_every and (index % progress_every == 0 or index == total):
                elapsed = time.time() - started_at
                rate = elapsed / index
                remaining = (total - index) * rate
                print(
                    f"[{index}/{total}] {spec.tag} iter={iteration} "
                    f"{unit.company[:24]}/{unit.code} -> {rating} "
                    f"| {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m left",
                    file=sys.stderr,
                    flush=True,
                )

    if unload_between_models and active_tag is not None:
        if unload_model(active_tag, base_url):
            unloaded.append(active_tag)
        print(f"unloaded {active_tag}", file=sys.stderr, flush=True)

    return {
        "stopped_early": stopped,
        "unloaded_models": unloaded,
        "requested_calls": total,
        "already_on_disk": len(completed),
        "written": counters["written"],
        "failed": counters["failed"],
        "retries": counters["retried"],
        # Attempts cut off at the deadline. High counts mean the stall mode is
        # active, not that the models are slow.
        "deadline_aborts": counters["deadline_aborts"],
        # A truncated answer parsed into a rating, but the model was cut off
        # mid-sentence, so the justification behind it is incomplete.
        "truncated_answers": counters["truncated"],
        "prompt_tokens": counters["prompt_tokens"],
        "completion_tokens": counters["completion_tokens"],
        "total_tokens": counters["prompt_tokens"] + counters["completion_tokens"],
        "elapsed_sec": round(time.time() - started_at, 1),
        "output": str(out_path),
    }
