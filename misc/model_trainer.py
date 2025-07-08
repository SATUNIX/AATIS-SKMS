#!/usr/bin/env python3
"""
model_trainer.py
──────────────────────
Self-RAG → SKMS refresh → Axolotl fine-tune
Stable for week-long CPU runs on ≥128 GB RAM.

• SELF-RAG harvests N pairs and APPENDS to data/selfrag_pairs.jsonl
• SKMS refreshes knowledge
• Hourly dedup merges seed + pairs → full_corpus.jsonl
• If ≥ MIN_NEW_PAIRS appear ⇒ Axolotl QLoRA-GoRA fine-tune
• Quick eval keeps best ckpt + last 3 for rollback
• Adaptive waits for CPU < LOAD% and RAM < 90 %

Install:
    pip install psutil axolotl selfrag
Clone SKMS repo and ensure `skms.py` is on PATH.
"""

import subprocess, pathlib, time, datetime, argparse, logging, sys, json, psutil, os
from hashlib import sha1

# ─────────────── configurable constants ──────────────── #
STEPS_PER_CYCLE  = 500
MIN_NEW_PAIRS    = 100
CPU_LOAD_LIMIT   = 80.0     # %
MAX_WAIT         = 3600     # sec
IDLE_SECONDS     = 10
KEEP_LATEST_N    = 3        # plus best
SAVE_CHECKPOINTS = True

PROJECT          = pathlib.Path(__file__).resolve().parent
DATA_DIR         = PROJECT / "data"
LOG_DIR          = PROJECT / "logs"
CKPT_DIR         = PROJECT / "checkpoints"
AXO_YAML         = PROJECT / "axolotl_job.yaml"
SEED_STATIC      = DATA_DIR / "seed_static.jsonl"
PAIR_JSON        = DATA_DIR / "selfrag_pairs.jsonl"
CORPUS_JSON      = DATA_DIR / "full_corpus.jsonl"

SELF_RAG_CMD     = "selfrag run --model mistral-7b --steps {steps} --topic \"{topic}\" --append_out {out}"
SKMS_REFRESH_CMD = "python skms.py refresh"
AXO_CMD          = f"axolotl run {AXO_YAML}"
EVAL_PROMPTS     = ["2+2?", "What is HTTP?", "Define entropy."]

# ───────────────────── directories & logging ───────────────────── #
for d in (DATA_DIR, LOG_DIR, CKPT_DIR):
    d.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"loop_{datetime.date.today()}.log"),
    ],
)
log = logging.getLogger("orchestrator")

# ───────────────────── utility helpers ───────────────────── #
def run(cmd: str):
    log.info("CMD %s", cmd)
    subprocess.run(cmd, shell=True, check=True)

def wait_resources(cpu_limit=CPU_LOAD_LIMIT, ram_limit=90.0):
    t0 = time.time()
    while True:
        if psutil.cpu_percent(interval=1) < cpu_limit and psutil.virtual_memory().percent < ram_limit:
            return
        if time.time() - t0 > MAX_WAIT:
            log.warning("Resource wait timed-out—continuing under load.")
            return
        time.sleep(5)

def dedup(lines):
    seen, out = set(), []
    for ln in lines:
        h = sha1(ln.encode()).hexdigest()
        if h not in seen:
            seen.add(h); out.append(ln)
    return out

def merge_corpus():
    """Dedup seed + SELF-RAG pairs into one JSONL. Return corpus size."""
    lines = []
    for p in (SEED_STATIC, PAIR_JSON):
        if p.exists():
            lines += p.read_text().splitlines()
    merged = dedup(lines)
    CORPUS_JSON.write_text("\n".join(merged) + ("\n" if merged else ""))
    return len(merged)

def ensure_yaml():
    if AXO_YAML.exists():
        return
    AXO_YAML.write_text(f"""
base_model: mistralai/Mistral-7B-Instruct-v0.2
dataset:
  path: {CORPUS_JSON}
llm_int8_enable_fp32_cpu_offload: true
qlora:
  bits: 4
  quant_type: nf4
adaptive_lora:
  enable: true
  grad_sample: 128
training:
  epochs: 1
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 16
  learning_rate: 2e-5
output_dir: {CKPT_DIR}
""".lstrip())
    log.info("Created Axolotl YAML.")

def quick_eval(model_path: pathlib.Path):
    import transformers, torch
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.eval()
    good = 0
    for p in EVAL_PROMPTS:
        ids = tok(p, return_tensors="pt").to("cpu")
        out = model.generate(**ids, max_new_tokens=16)
        ans = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        if any(k in ans.lower() for k in ["4", "protocol", "state"]):
            good += 1
    score = good / len(EVAL_PROMPTS)
    log.info("Eval %.2f for %s", score, model_path.name)
    return score

def prune_checkpoints(best_path: pathlib.Path):
    ckpts = sorted(CKPT_DIR.glob("*"), key=os.path.getmtime, reverse=True)
    keep = {best_path} | set(ckpts[:KEEP_LATEST_N])
    for p in ckpts:
        if p not in keep:
            shutil.rmtree(p, ignore_errors=True)
            log.info("Pruned %s", p.name)

# ─────────────────────────── main loop ─────────────────────────── #
def main(topic: str):
    ensure_yaml()
    best_score = -1
    best_path  = None
    corpus_size = merge_corpus()
    last_merge  = time.time()  # hourly dedup

    while True:
        start_cycle = time.time()
        log.info("── New cycle for topic: %s", topic)

        # SELF-RAG harvest
        wait_resources(); run(SELF_RAG_CMD.format(steps=STEPS_PER_CYCLE, topic=topic, out=PAIR_JSON))

        # SKMS refresh
        wait_resources(); run(SKMS_REFRESH_CMD)

        # Merge corpus hourly
        if time.time() - last_merge > 3600:
            corpus_size = merge_corpus()
            last_merge  = time.time()

        # count SELF-RAG additions
        new_pairs = corpus_size - merge_corpus()  # merge again to count
        corpus_size += new_pairs
        log.info("New pairs: %d  | corpus: %d", new_pairs, corpus_size)

        if new_pairs >= MIN_NEW_PAIRS:
            try:
                wait_resources(); run(AXO_CMD)
            except subprocess.CalledProcessError:
                run("pkill -f axolotl")  # kill stray workers
                continue

            latest_ckpt = max(CKPT_DIR.glob("*"), key=os.path.getmtime)
            score = quick_eval(latest_ckpt)
            if score > best_score:
                best_score, best_path = score, latest_ckpt
                log.info("NEW BEST %.2f at %s", score, best_path.name)

            if SAVE_CHECKPOINTS:
                prune_checkpoints(best_path)

        elapsed = time.time() - start_cycle
        log.info("Cycle done in %.1f min — idle %d s", elapsed/60, IDLE_SECONDS)
        time.sleep(IDLE_SECONDS)

# ────────────────────────── CLI entry ─────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="Domain topic (e.g. 'quantum networking')")
    args = ap.parse_args()
    try:
        main(args.topic)
    except KeyboardInterrupt:
        log.info("Interrupted — exiting.")