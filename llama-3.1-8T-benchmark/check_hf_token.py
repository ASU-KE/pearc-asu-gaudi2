#!/usr/bin/env python3
"""
check_hf_token.py — diagnose the HuggingFace 401 / gated-repo error.

Run it the SAME way the benchmark runs so it sees the same token/env:

  # bare (mamba) env:
  HF_HOME=/scratch/tianche5/huggingface python check_hf_token.py

  # inside the apptainer image used by the job:
  apptainer exec --nv \
    --env HF_HOME="$HF_HOME" \
    --env HUGGINGFACE_HUB_TOKEN="$(cat $HF_HOME/token)" \
    <image.sif> python /path/to/check_hf_token.py

It reports, in order: (1) which token it found, (2) whether it authenticates
(whoami), (3) whether the account has gated access to each model the benchmark
uses. Each line is PASS or FAIL with the exact next step.
"""

import os
import sys

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
]


def find_token():
    """Same precedence the HF stack uses: env vars, then $HF_HOME/token, then
    the default cache location."""
    for env in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env].strip(), f"env:{env}"
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for path in (os.path.join(hf_home, "token"),
                 os.path.expanduser("~/.cache/huggingface/token")):
        if os.path.isfile(path):
            tok = open(path).read().strip()
            if tok:
                return tok, f"file:{path}"
    return None, None


def gated_like(exc):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    msg = str(exc).lower()
    keys = ("gated", "restricted", "awaiting", "access to model", "must have access")
    return status in (401, 403) or any(k in msg for k in keys), status


def main():
    try:
        import huggingface_hub
        from huggingface_hub import whoami, HfApi, hf_hub_download
    except Exception as e:
        print(f"FAIL: cannot import huggingface_hub ({e})")
        sys.exit(2)

    print(f"huggingface_hub {huggingface_hub.__version__}")
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(unset)')}")

    tok, src = find_token()
    if not tok:
        print("FAIL: no token found in env (HF_TOKEN/HUGGINGFACE_HUB_TOKEN) or $HF_HOME/token.")
        print("  -> export HUGGINGFACE_HUB_TOKEN or place the token at $HF_HOME/token.")
        sys.exit(1)
    print(f"token: {src}  (len={len(tok)}, prefix={tok[:3]}…)")

    # 1) does the token authenticate at all?
    try:
        account = whoami(token=tok).get("name")
        print(f"PASS: authenticated as '{account}' (token is valid)")
    except Exception as e:
        print(f"FAIL: token did not authenticate -> {type(e).__name__}: {e}")
        print("  -> token is invalid/expired or not passed in. Make a READ token at "
              "https://huggingface.co/settings/tokens")
        sys.exit(1)

    # 2) gated access. model_info only checks metadata visibility; the
    #    authoritative test is downloading a gated FILE, which is exactly what
    #    vLLM does (and can 403 even when model_info succeeds).
    api = HfApi()
    bad = False
    for m in MODELS:
        try:
            api.model_info(m, token=tok)
            meta = "ok"
        except Exception as e:
            _, s = gated_like(e)
            meta = f"FAIL(HTTP {s})"
        try:
            hf_hub_download(m, "config.json", token=tok)
            print(f"PASS: file download OK -> {m}   (model_info: {meta})")
        except Exception as e:
            bad = True
            is_gated, status = gated_like(e)
            tag = "gated / NOT authorized" if is_gated else type(e).__name__
            print(f"FAIL: cannot download files (HTTP {status}; {tag}) -> {m}   (model_info: {meta})")
            print(f"  -> on https://huggingface.co/{m} as '{account}', confirm it says "
                  f"'You have been granted access' (not pending/awaiting review), AND use a "
                  f"CLASSIC read token (fine-grained tokens need 'Read access to public gated repos').")

    print("\nSUMMARY:", "all good ✓" if not bad
          else "token works but at least one model is not accessible — see FAIL lines above.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
