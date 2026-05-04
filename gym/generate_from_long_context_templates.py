"""
Template-based generation of long context retrieval samples.

This generator constructs long-context retrieval tasks of two kinds.

  Word-list tasks (procedurally generated word lists):
  - common_words:          Find the top-K most frequent words.
  - rare_words:            Find the top-K least frequent words.
  - count_word:            Count occurrences of a specific word.
  - word_at_position:      Identify the word at a numbered position.
  - frequency_comparison:  Compare frequency of two given words.

  Haystack tasks (needle-in-a-haystack over documents):
  - needle_single_number:          One number hidden in document for one key.
  - needle_multi_number_same_key:  Multiple numbers for same key scattered in document.
  - needle_multi_number_diff_keys: Numbers for different keys interleaved in document.
  - needle_uuid:                   UUID key-value pairs embedded in/around document text.

Sizing parameters:
  Word-list tasks use --num_context_words (approximate total words in list, min 20).
  Haystack tasks use --max_seq_length + --tokenizer (target context in tokens).

Usage examples:
    # Word-list tasks only (5 types x 2000 = 10000 samples)
    python generate_from_long_context_templates.py \
        --output_file word_list.jsonl \
        --num_samples 2000 \
        --num_context_words 50 \
        --task_types common_words rare_words count_word \
                     word_at_position frequency_comparison

    # Word-list tasks at multiple context sizes (5 types x 3 sizes x 2000 = 30000 samples)
    python generate_from_long_context_templates.py \
        --output_file word_list.jsonl \
        --num_samples 2000 \
        --num_context_words 50 100 200 \
        --task_types common_words rare_words count_word \
                     word_at_position frequency_comparison

    # Haystack tasks only (4 types x 1 seq_len x 2000 = 8000 samples)
    python generate_from_long_context_templates.py \
        --output_file haystack.jsonl \
        --num_samples 2000 \
        --max_seq_length 4096 \
        --tokenizer Polygl0t/Tucano2-0.6B-Base \
        --docs_dir assets \
        --task_types needle_single_number needle_multi_number_same_key \
                     needle_multi_number_diff_keys needle_uuid

    # Haystack tasks at multiple sequence lengths (4 types x 3 lengths x 2000 = 24000 samples)
    python generate_from_long_context_templates.py \
        --output_file haystack.jsonl \
        --num_samples 2000 \
        --max_seq_length 2048 4096 8192 \
        --tokenizer Polygl0t/Tucano2-0.6B-Base \
        --docs_dir assets \
        --task_types needle_single_number needle_multi_number_same_key \
                     needle_multi_number_diff_keys needle_uuid

    # All tasks (word-list sized by words, haystack sized by tokens)
    python generate_from_long_context_templates.py \
        --output_file long_tasks.jsonl \
        --num_samples 1000 \
        --num_context_words 50 100 200 400 \
        --max_seq_length 1024 2048 4096 8192 \
        --tokenizer Polygl0t/Tucano2-0.6B-Base \
        --docs_dir assets
"""

import json
import uuid
import random
import re
import sys
import hashlib
import argparse
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

from utils import SUBSTANTIVOS, ADJETIVOS, VERBOS
from long_context_templates import (
    LONG_CONTEXT_TEMPLATES,
    LONG_CONTEXT_TASK_TYPES,
    WORD_LIST_TASK_TYPES,
    HAYSTACK_TASK_TYPES,
)
from tasks_metadata import LONG_CONTEXT_DEFAULTS, HAYSTACK_DEFAULTS


# Verifier ID mapping 
TASK_TYPE_TO_VERIFIER = {
    "common_words": "long_context:common_words",
    "rare_words": "long_context:rare_words",
    "count_word": "long_context:count_word",
    "word_at_position": "long_context:word_at_position",
    "frequency_comparison": "long_context:frequency_comparison",
    "needle_single_number": "haystack:needle_single_number",
    "needle_multi_number_same_key": "haystack:needle_multi_number_same_key",
    "needle_multi_number_diff_keys": "haystack:needle_multi_number_diff_keys",
    "needle_uuid": "haystack:needle_uuid",
}


#  Word bank
_BANK_RNG = random.Random(42)
WORDS = sorted(set(SUBSTANTIVOS + ADJETIVOS + VERBOS))
_BANK_RNG.shuffle(WORDS)

# Tokens reserved for the answer, per task type
_HS_TOKENS_TO_GENERATE = HAYSTACK_DEFAULTS["tokens_to_generate"]


#  Word-list helpers
def build_word_list(num_unique, common_repeats, uncommon_repeats, common_nums):
    """Build a shuffled numbered word list with controlled frequency distribution.

    Returns (word_list, common_words, uncommon_words, frequency_counter).
    """
    num_unique = min(num_unique, len(WORDS))
    common_nums = min(common_nums, max(num_unique - 2, 1))

    pool = random.sample(WORDS, num_unique)
    common = pool[:common_nums]
    uncommon = pool[common_nums:]

    word_list = common * common_repeats + uncommon * uncommon_repeats
    random.shuffle(word_list)

    return word_list, common, uncommon, Counter(word_list)


def format_numbered_list(word_list):
    """Format as '1. word1 2. word2 3. word3 ...'."""
    return " ".join(f"{i + 1}. {w}" for i, w in enumerate(word_list))


#  Document / haystack helpers
def load_documents(docs_dir):
    """Load all .txt documents from *docs_dir*, sorted by name."""
    docs_path = Path(docs_dir)
    files = sorted(docs_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")
    documents = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)
    return documents


def get_document_chunk(documents, num_chars, rng):
    """Return a contiguous chunk of *num_chars* characters from a random document."""
    doc = rng.choice(documents)
    if len(doc) <= num_chars:
        return doc
    start = rng.randint(0, len(doc) - num_chars)
    return doc[start : start + num_chars]


def make_compound_key(rng):
    """Generate a compound key like 'seco-dia' from word banks."""
    adj = rng.choice(ADJETIVOS)
    noun = rng.choice(SUBSTANTIVOS)
    return f"{adj}-{noun}"


def make_unique_compound_keys(n, rng):
    """Generate *n* unique compound keys."""
    keys = set()
    while len(keys) < n:
        keys.add(make_compound_key(rng))
    return list(keys)


def make_random_number(rng, low=1_000_000, high=9_999_999):
    """Generate a random 7-digit integer."""
    return rng.randint(low, high)


def make_uuid(rng):
    """Generate a random UUID4 string using the given RNG for reproducibility."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _split_into_sentences(text):
    """Split text on sentence-ending punctuation, keeping delimiters."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p.strip()]


def insert_needles_distributed(haystack, needles, rng):
    """Insert needle strings at random positions throughout the haystack text.

    Splits the haystack into sentence-like chunks and inserts needles between them.
    Returns `(text, inserted_needles)` where `inserted_needles` is the list of
    needles that were actually placed in the text (in original order). All needles
    are guaranteed to be inserted: if there are more needles than gap positions,
    overflow needles are appended at the end of the text.
    """
    sentences = _split_into_sentences(haystack)
    if len(sentences) < 2:
        return " ".join(needles) + " " + haystack, list(needles)

    n = len(needles)
    if n >= len(sentences):
        # Place one needle in each available gap; remaining needles will be
        # appended at the end so all needles are present in the final text.
        primary = needles[: len(sentences)]
        overflow = needles[len(sentences) :]
        positions = list(range(len(primary)))
    else:
        primary = list(needles)
        overflow = []
        step = len(sentences) / (n + 1)
        positions = [int(step * (i + 1)) for i in range(n)]
        positions = [
            max(1, min(p + rng.randint(-1, 1), len(sentences) - 1))
            for p in positions
        ]

    # Sort descending so insertions don't shift earlier positions.
    pairs = sorted(zip(primary, positions), key=lambda x: x[1], reverse=True)
    for needle, pos in pairs:
        sentences.insert(pos, needle)

    text = " ".join(sentences)
    if overflow:
        text = text + " " + " ".join(overflow)
    return text, list(needles)


def insert_needles_at_start(haystack, needles):
    """Insert all needle strings at the beginning of the haystack.

    Returns `(text, inserted_needles)` for parity with
    :func:`insert_needles_distributed`.
    """
    return " ".join(needles) + " " + haystack, list(needles)


#  Prompt assembly
def assemble_prompt(template, context, **fmt):
    """Combine a random preamble + context + a random question."""
    preamble = random.choice(template["preambles"])
    question = random.choice(template["questions"]).format(**fmt)
    return f"{preamble}\n{context}\nPergunta: {question}"


#  Word-list verifier kwargs generation 
def generate_verifier_kwargs(task_type, word_list, freq, common, **kw):
    """Produce verifier_kwargs for a word-list task type."""

    if task_type == "common_words":
        top_k = kw["top_k"]
        ranked = sorted(common, key=lambda w: (-freq[w], w))[:top_k]
        return {"expected_words": ranked}

    if task_type == "rare_words":
        top_k = kw["top_k"]
        by_freq = sorted(freq.items(), key=lambda x: (x[1], x[0]))
        rarest = [w for w, _ in by_freq[:top_k]]
        return {"expected_words": rarest}

    if task_type == "count_word":
        target = kw["target_word"]
        count = freq.get(target, 0)
        return {"target_word": target, "expected_count": count}

    if task_type == "word_at_position":
        pos = kw["position"]
        word = word_list[pos - 1]
        return {"position": pos, "expected_word": word}

    if task_type == "frequency_comparison":
        wa, wb = kw["word_a"], kw["word_b"]
        ca, cb = freq.get(wa, 0), freq.get(wb, 0)
        winner = wa if ca >= cb else wb
        return {"word_a": wa, "word_b": wb, "expected_winner": winner}

    return {}


#  Haystack sample builders
def _build_needle_single_number(template, documents, num_chars, rng):
    """Build: one number hidden in document for one key."""
    key = make_compound_key(rng)
    value = make_random_number(rng)

    needle_fmt = rng.choice(template["needle_formats"])
    needle = needle_fmt.format(key=key, value=value)

    haystack = get_document_chunk(documents, num_chars, rng)
    text, _ = insert_needles_distributed(haystack, [needle], rng)

    preamble = rng.choice(template["preambles"])
    question = rng.choice(template["questions"]).format(key=key)
    prompt = f"{preamble}\n{text}\nPergunta: {question}"

    verifier_kwargs = {"key": key, "expected_values": {key: [str(value)]}}
    return prompt, verifier_kwargs


def _build_needle_multi_number_same_key(template, documents, num_chars, rng):
    """Build: multiple numbers for the same key scattered in document."""
    defaults = HAYSTACK_DEFAULTS
    num_needles = rng.randint(*defaults["num_needles_range"])

    key = make_compound_key(rng)
    values = [make_random_number(rng) for _ in range(num_needles)]

    needle_fmt = rng.choice(template["needle_formats"])
    needles = [needle_fmt.format(key=key, value=v) for v in values]
    # Map needle -> value so we can rebuild expected_values from what
    # actually got inserted.
    needle_to_value = dict(zip(needles, values))

    haystack = get_document_chunk(documents, num_chars, rng)

    if rng.random() < 0.5:
        text, inserted = insert_needles_at_start(haystack, needles)
    else:
        text, inserted = insert_needles_distributed(haystack, needles, rng)

    inserted_values = [needle_to_value[n] for n in inserted if n in needle_to_value]

    preamble = rng.choice(template["preambles"])
    question = rng.choice(template["questions"]).format(key=key)
    prompt = f"{preamble}\n{text}\nPergunta: {question}"

    verifier_kwargs = {
        "key": key,
        "expected_values": {key: [str(v) for v in inserted_values]},
    }
    return prompt, verifier_kwargs


def _build_needle_multi_number_diff_keys(template, documents, num_chars, rng):
    """Build: one number per key for several different keys, interleaved."""
    defaults = HAYSTACK_DEFAULTS
    num_keys = rng.randint(*defaults["num_keys_range"])

    keys = make_unique_compound_keys(num_keys, rng)
    kv_pairs = {k: str(make_random_number(rng)) for k in keys}

    needle_fmt = rng.choice(template["needle_formats"])
    key_order = list(kv_pairs.keys())
    needles = [needle_fmt.format(key=k, value=kv_pairs[k]) for k in key_order]
    needle_to_key = dict(zip(needles, key_order))

    haystack = get_document_chunk(documents, num_chars, rng)
    text, inserted = insert_needles_distributed(haystack, needles, rng)

    inserted_keys = [needle_to_key[n] for n in inserted if n in needle_to_key]

    keys_str = ", ".join(inserted_keys)
    preamble = rng.choice(template["preambles"])
    question = rng.choice(template["questions"]).format(keys_str=keys_str)
    prompt = f"{preamble}\n{text}\nPergunta: {question}"

    verifier_kwargs = {
        "expected_values": {k: [kv_pairs[k]] for k in inserted_keys},
    }
    return prompt, verifier_kwargs


def _build_needle_uuid(template, documents, num_chars, rng):
    """Build: UUID key->value pairs, query one key."""
    defaults = HAYSTACK_DEFAULTS
    num_pairs = rng.randint(*defaults["num_uuid_pairs_range"])

    pairs = {}
    for _ in range(num_pairs):
        k = make_uuid(rng)
        v = make_uuid(rng)
        pairs[k] = v

    needle_fmt = rng.choice(template["needle_formats"])
    key_order = list(pairs.keys())
    needles = [needle_fmt.format(key=k, value=pairs[k]) for k in key_order]
    needle_to_key = dict(zip(needles, key_order))

    if rng.random() < 0.5 and documents:
        haystack = get_document_chunk(documents, num_chars, rng)
        text, inserted = insert_needles_distributed(haystack, needles, rng)
    else:
        text = " ".join(needles)
        inserted = list(needles)

    inserted_keys = [needle_to_key[n] for n in inserted if n in needle_to_key]
    if not inserted_keys:
        # Fallback: should be unreachable, but never query a key not in text.
        inserted_keys = key_order

    query_key = rng.choice(inserted_keys)
    expected_value = pairs[query_key]

    preamble = rng.choice(template["preambles"])
    question = rng.choice(template["questions"]).format(query_key=query_key)
    prompt = f"{preamble}\n{text}\nPergunta: {question}"

    verifier_kwargs = {"query_key": query_key, "expected_values": {query_key: [expected_value]}}
    return prompt, verifier_kwargs


# Haystack dispatch table
_HAYSTACK_BUILDERS = {
    "needle_single_number": _build_needle_single_number,
    "needle_multi_number_same_key": _build_needle_multi_number_same_key,
    "needle_multi_number_diff_keys": _build_needle_multi_number_diff_keys,
    "needle_uuid": _build_needle_uuid,
}


#  Token-based calibration 
def calibrate_num_chars(tokenizer, max_seq_length, task_type, documents, rng, step=500):
    """Grow the document chunk size until the prompt fills the token budget (haystack tasks)."""
    tokens_reserve = _HS_TOKENS_TO_GENERATE.get(task_type, 200)
    template = next(t for t in LONG_CONTEXT_TEMPLATES if t["task_type"] == task_type)

    best = step
    num_chars = step

    while num_chars <= max(len(d) for d in documents):
        local_rng = random.Random(rng.randint(0, 2**31))
        try:
            sample = build_sample(
                template, documents=documents,
                num_chars=num_chars, rng=local_rng,
            )
        except Exception:
            break

        prompt = sample["prompt"]
        tok_len = len(tokenizer(prompt).input_ids) + tokens_reserve
        if tok_len > max_seq_length:
            break
        best = num_chars
        num_chars += step

    return best


#  Unified sample construction
def build_sample(template, *, num_words=None,
                 documents=None, num_chars=None, rng=None):
    """Build one complete retrieval sample in the standardized format.

    For word-list tasks, provide *num_words* (target total list length).
    For haystack tasks, provide *documents*, *num_chars*, and *rng*.

    Returns a dict with keys: id, prompt, verifier_id_list, kwargs.
    """
    task_type = template["task_type"]
    verifier_id = TASK_TYPE_TO_VERIFIER[task_type]

    if task_type in WORD_LIST_TASK_TYPES:
        prompt, verifier_kwargs = _build_word_list(
            template, task_type, num_words,
        )
    elif task_type in HAYSTACK_TASK_TYPES:
        builder = _HAYSTACK_BUILDERS[task_type]
        prompt, verifier_kwargs = builder(
            template, documents, num_chars, rng,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    sample_id = hashlib.md5(prompt.encode()).hexdigest()
    return {
        "id": sample_id,
        "prompt": prompt,
        "verifier_id_list": [verifier_id],
        "kwargs": [json.dumps(verifier_kwargs, ensure_ascii=False)],
    }


# top_k range adapts to the target list size
_TOP_K_RANGES = [
    (50,  (1, 2)),   # 20-50 words
    (100, (1, 5)),   # 50-100 words
]  # 100+ words: use defaults["top_k_range"]


def _top_k_range_for(num_words, defaults):
    """Return the (lo, hi) top_k range appropriate for *num_words*."""
    for threshold, rng in _TOP_K_RANGES:
        if num_words <= threshold:
            return rng
    return defaults["top_k_range"]


def _build_word_list(template, task_type, num_words):
    """Build a word-list retrieval sample. Returns (prompt, verifier_kwargs).

    *num_words* is the approximate target total number of words in the list.
    The function derives the vocabulary pool size and repeat counts to hit
    this target while preserving a frequency gap between common and uncommon
    words.

    The number of top repeating words asked for (top_k) scales with list size:
      20-50 words  -> top_k in 1..2
      50-100 words -> top_k in 1..5
      100+ words   -> top_k in defaults["top_k_range"]
    """
    defaults = LONG_CONTEXT_DEFAULTS
    top_k = random.randint(*_top_k_range_for(num_words, defaults))
    common_repeats = random.randint(*defaults["common_repeats_range"])
    uncommon_repeats = random.randint(*defaults["uncommon_repeats_range"])
    common_nums = random.randint(*defaults["common_nums_range"])

    if task_type in ("common_words", "rare_words"):
        common_nums = max(common_nums, top_k)

    # Derive num_unique (pool size) so that
    #   total ≈ common_nums * cr + (num_unique - common_nums) * ur ≈ num_words
    # If the target is too small for the drawn repeat counts, halve them until
    # a viable pool size emerges.
    num_unique = (num_words - common_nums * (common_repeats - uncommon_repeats)) // max(uncommon_repeats, 1)
    while num_unique < common_nums + 2 and common_repeats > 2:
        common_repeats = max(2, common_repeats // 2)
        uncommon_repeats = max(1, uncommon_repeats // 2)
        num_unique = (num_words - common_nums * (common_repeats - uncommon_repeats)) // max(uncommon_repeats, 1)

    num_unique = max(common_nums + 2, min(num_unique, len(WORDS)))
    common_nums = min(common_nums, max(num_unique - 2, 1))
    common_nums = max(common_nums, 3)

    word_list, common, uncommon, freq = build_word_list(
        num_unique, common_repeats, uncommon_repeats, common_nums
    )

    # Truncate to the target if the list overshot
    if len(word_list) > num_words:
        word_list = word_list[:num_words]
        freq = Counter(word_list)
        common = [w for w in common if freq[w] > 0]
        uncommon = [w for w in uncommon if freq[w] > 0]

    # For rare_words: ensure exactly top_k words are uniquely the rarest.
    # Without this, many words may share the minimum frequency, making the
    # answer ambiguous (any subset of tied words would be correct).
    if task_type == "rare_words":
        min_freq = min(freq.values())
        at_min = [w for w in freq if freq[w] == min_freq]

        if len(at_min) > top_k:
            # Designate top_k words as the rarest; boost all others above min_freq
            rarest_set = set(random.sample(at_min, top_k))
            for w in at_min:
                if w not in rarest_set:
                    word_list.append(w)

            random.shuffle(word_list)
            freq = Counter(word_list)
            uncommon = [w for w in uncommon if freq[w] > 0]

    # For common_words: avoid ties at the top_k boundary. If the count of the
    # word at rank top_k equals the count of the word at rank top_k+1, the
    # answer is ambiguous (multiple equally-valid sets of "top_k most common"
    # words exist). Boost the chosen top_k words by one occurrence each to
    # break the tie deterministically.
    if task_type == "common_words":
        ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        if len(ranked) > top_k:
            kth_count = ranked[top_k - 1][1]
            next_count = ranked[top_k][1]
            if kth_count == next_count:
                top_words = [w for w, _ in ranked[:top_k]]
                word_list.extend(top_words)
                random.shuffle(word_list)
                freq = Counter(word_list)
                common = [w for w in common if freq[w] > 0]

    context = format_numbered_list(word_list)

    # Task-specific format kwargs
    fmt = {}
    vkw = {}

    if task_type == "common_words":
        fmt["top_k"] = top_k
        vkw = {"top_k": top_k}

    elif task_type == "rare_words":
        fmt["top_k"] = top_k
        vkw = {"top_k": top_k}

    elif task_type == "count_word":
        target = random.choice(common + uncommon)
        fmt["target_word"] = target
        vkw = {"target_word": target}

    elif task_type == "word_at_position":
        pos = random.randint(1, len(word_list))
        fmt["position"] = pos
        vkw = {"position": pos}

    elif task_type == "frequency_comparison":
        wa = random.choice(common) if common else random.choice(list(freq.keys()))
        others = [w for w in (uncommon if uncommon else list(freq.keys())) if w != wa]
        wb = random.choice(others) if others else wa
        fmt["word_a"] = wa
        fmt["word_b"] = wb
        vkw = {"word_a": wa, "word_b": wb}

    prompt = assemble_prompt(template, context, **fmt)
    verifier_kwargs = generate_verifier_kwargs(
        task_type, word_list, freq, common, **vkw
    )

    return prompt, verifier_kwargs


#  Validation 
def validate_sample(sample):
    """Return a list of issues (empty means valid)."""
    issues = []
    if not sample.get("prompt", "").strip():
        issues.append("Empty prompt")
    if not sample.get("verifier_id_list"):
        issues.append("Empty verifier_id_list")
    if len(sample.get("verifier_id_list", [])) != len(sample.get("kwargs", [])):
        issues.append("verifier_id_list and kwargs length mismatch")
    return issues


#  Main 
def main(args):
    output_path = Path(args.output_file)
    seed = args.seed
    max_retries = 20

    # Load tokenizer (optional)
    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True,
            cache_dir=args.cache_dir,
        )

    # Filter templates by requested task types
    templates = LONG_CONTEXT_TEMPLATES
    if args.task_types:
        templates = [t for t in templates if t["task_type"] in args.task_types]
        if not templates:
            raise ValueError(
                f"No templates match task_types={args.task_types}. "
                f"Available: {LONG_CONTEXT_TASK_TYPES}"
            )

    # Separate templates by category
    wl_templates = [t for t in templates if t["task_type"] in WORD_LIST_TASK_TYPES]
    hs_templates = [t for t in templates if t["task_type"] in HAYSTACK_TASK_TYPES]

    # Load documents if haystack tasks are requested
    documents = None
    if hs_templates:
        if not args.docs_dir:
            raise ValueError(
                "Haystack task types require --docs_dir. "
                f"Haystack types requested: {[t['task_type'] for t in hs_templates]}"
            )
        documents = load_documents(args.docs_dir)
        print(f"Loaded {len(documents)} documents from {args.docs_dir}")

    # Build the generation plan: list of (template, build_kwargs, context_size)
    generation_plan = []  # [(template, build_kwargs_dict, context_size_value)]
    rng = random.Random(seed)

    # Word-list tasks: always sized by --num_context_words
    if wl_templates:
        if not args._num_context_words_explicit:
            print(
                f"Warning: --num_context_words not provided for word-list tasks. "
                f"Using default {args.num_context_words}."
            )
        for target_words in args.num_context_words:
            target_words = max(target_words, 20)
            print(f"Target ~{target_words} words per list")
            for t in wl_templates:
                generation_plan.append((t, {"num_words": target_words}, target_words))

    # Haystack tasks: always sized by --max_seq_length + --tokenizer
    if hs_templates:
        if not tokenizer:
            raise ValueError(
                "Haystack tasks require --tokenizer and --max_seq_length. "
                f"Haystack types requested: {[t['task_type'] for t in hs_templates]}"
            )
        if not args._max_seq_length_explicit:
            print(
                f"Warning: --max_seq_length not provided for haystack tasks. "
                f"Using default {args.max_seq_length}."
            )
        for seq_len in args.max_seq_length:
            cc = {}
            print(f"Calibrating context sizes for max_seq_length={seq_len}...")
            for t in hs_templates:
                cal_rng = random.Random(seed)
                nc = calibrate_num_chars(
                    tokenizer, seq_len, t["task_type"], documents, cal_rng
                )
                cc[t["task_type"]] = nc
                print(f"  {t['task_type']}: {nc} chars")
            for t in hs_templates:
                generation_plan.append(
                    (t, {"num_chars": cc[t["task_type"]]}, seq_len)
                )

    num_sizes = len({ctx_size for _, _, ctx_size in generation_plan})
    print(
        f"Generating {args.num_samples} samples per task type per size, "
        f"{len(templates)} task type(s), {num_sizes} size(s) (seed={seed})"
    )

    # Generate samples
    samples = []
    seen = set()
    sample_counter = 0
    retries_used = 0
    type_dist = Counter()
    size_dist = Counter()

    for template, build_kwargs, ctx_size in generation_plan:
        tt = template["task_type"]

        for i in range(args.num_samples):
            sample = None

            for attempt in range(max_retries):
                random.seed(seed + sample_counter * max_retries + attempt)

                if tt in WORD_LIST_TASK_TYPES:
                    candidate = build_sample(
                        template,
                        num_words=build_kwargs["num_words"],
                    )
                else:
                    sample_rng = random.Random(rng.randint(0, 2**63))
                    candidate = build_sample(
                        template,
                        documents=documents,
                        num_chars=build_kwargs["num_chars"],
                        rng=sample_rng,
                    )

                sid = candidate["id"]
                if sid not in seen:
                    sample = candidate
                    seen.add(sid)
                    break

                retries_used += 1

            if sample is None:
                print(f"  Warning: could not produce unique sample #{i + 1} for {tt} (size={ctx_size})")
                continue

            samples.append(sample)
            type_dist[tt] += 1
            size_dist[ctx_size] += 1
            sample_counter += 1

    # Validate
    total_issues = 0
    for s in samples:
        issues = validate_sample(s)
        if issues:
            total_issues += len(issues)
            if args.verbose:
                print(f"  ID {s['id']}: {issues}")

    unique_count = len({s["id"] for s in samples})

    print(f"\nResults:")
    print(f"  Generated samples:  {len(samples)}")
    print(
        f"  Unique:             {unique_count}/{len(samples)}"
        f"  ({100 * unique_count / max(len(samples), 1):.1f}%)"
    )
    print(f"  Validation issues:  {total_issues}")
    print(f"  Task distribution:  {dict(type_dist)}")
    print(f"  Size distribution:  {dict(size_dist)}")
    if retries_used:
        print(f"  Uniqueness retries: {retries_used}")

    assert total_issues == 0, f"FAIL: {total_issues} validation issues found"

    # Write output
    use_jsonl = output_path.suffix.lower() == ".jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if use_jsonl:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    fmt = "JSONL" if use_jsonl else "JSON"
    print(f"\nOutput written to: {output_path} ({fmt})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Template-based long context retrieval task generation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSON/JSONL file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to generate per task type (default: 500).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        nargs="+",
        default=[512],
        help=(
            "Target context length(s) in tokens for haystack tasks (requires --tokenizer). "
            "Pass multiple values to generate num_samples for each length (default: 512)."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name/path (required for haystack tasks).",
    )
    parser.add_argument(
        "--num_context_words",
        type=int,
        nargs="+",
        default=[50],
        help=(
            "Approximate total number of words in the generated word list "
            "(word-list tasks, minimum: 20). "
            "Pass multiple values to generate num_samples for each size (default: 50)."
        ),
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="./assets",
        help="Directory containing .txt documents (required for haystack tasks).",
    )
    parser.add_argument(
        "--task_types",
        nargs="+",
        default=None,
        choices=LONG_CONTEXT_TASK_TYPES,
        help="Restrict generation to these task types (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation info.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./.cache",
        help="HuggingFace cache directory for tokenizer models.",
    )

    args = parser.parse_args()
    args._num_context_words_explicit = "--num_context_words" in sys.argv
    args._max_seq_length_explicit = "--max_seq_length" in sys.argv
    main(args)
