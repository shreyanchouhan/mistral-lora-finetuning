"""Generate a 10k-pair technical-reasoning instruction dataset.

Output:
  data/train.jsonl  (10000 records)
  data/eval.jsonl   (200 records, held out)

Each record: {"instruction": str, "input": str, "output": str}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).parent
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"

# ---------- Topic templates ----------

DSA_TOPICS = [
    ("binary search", "O(log n) time on a sorted array; halve the search space each step."),
    ("merge sort", "Divide-and-conquer, O(n log n) time, O(n) extra space; stable."),
    ("quicksort", "Pivot partitioning, O(n log n) average / O(n^2) worst; in-place."),
    ("BFS", "Level-order graph traversal with a queue; finds shortest path in unweighted graphs."),
    ("DFS", "Depth-first traversal using a stack/recursion; useful for cycle detection and topological sort."),
    ("Dijkstra", "Single-source shortest paths on non-negative-weight graphs using a min-heap."),
    ("Kruskal", "Greedy MST algorithm using union-find on edges sorted by weight."),
    ("Prim", "Greedy MST grown from a starting vertex using a priority queue."),
    ("topological sort", "Linear order of a DAG using DFS post-order or Kahn's algorithm."),
    ("Floyd–Warshall", "All-pairs shortest paths via DP in O(V^3)."),
    ("Bellman–Ford", "Single-source shortest paths allowing negative edges; detects negative cycles."),
    ("KMP", "Linear-time string matching using a prefix-function failure table."),
    ("Rabin–Karp", "Rolling-hash string matching with average O(n+m)."),
    ("trie", "Prefix tree for fast set/prefix lookups over strings."),
    ("segment tree", "Range-query / point-update structure in O(log n) per op."),
    ("fenwick tree", "Binary indexed tree for prefix sums in O(log n)."),
    ("union-find", "Near-O(1) amortized merge/find with path compression and union by rank."),
    ("dynamic programming", "Memoize overlapping subproblems; bottom-up or top-down."),
    ("knapsack 0/1", "DP over items × capacity in O(nW) time/space."),
    ("longest common subsequence", "DP table over two strings in O(nm)."),
    ("longest increasing subsequence", "O(n log n) via patience-sort tails array."),
    ("heap", "Complete binary tree with heap order; O(log n) insert/extract."),
    ("hashmap", "Average O(1) lookup via hashing with collision resolution."),
    ("sliding window", "Two pointers tracking a contiguous range to amortize work."),
    ("two pointers", "Coordinated indices to scan sorted data in linear time."),
    ("monotonic stack", "Stack maintained in sorted order to answer next-greater/smaller queries."),
    ("bit manipulation", "XOR/AND/shifts for compact set operations and parity tricks."),
]

ML_TOPICS = [
    ("gradient descent", "Iteratively step in the direction of the negative gradient to minimize a loss."),
    ("backpropagation", "Chain-rule application that computes gradients layer by layer in a neural network."),
    ("dropout", "Randomly zero activations during training to reduce co-adaptation and overfitting."),
    ("batch normalization", "Normalize activations within a batch to stabilize and accelerate training."),
    ("layer normalization", "Normalize across features per sample; common in Transformers."),
    ("attention", "Weighted aggregation of values using softmax(QK^T / sqrt(d)) similarity."),
    ("transformer", "Stack of self-attention + feed-forward blocks with residual connections."),
    ("LoRA", "Low-rank adapter matrices A and B inserted into linear layers; only A, B are trained."),
    ("QLoRA", "Quantize the base model to 4-bit (NF4) and train LoRA adapters on top."),
    ("BPE tokenization", "Iteratively merge most-frequent byte pairs to form a sub-word vocabulary."),
    ("softmax", "Maps a real vector to a probability distribution using exponentials."),
    ("cross-entropy loss", "−Σ y log p̂; standard classification objective."),
    ("Adam optimizer", "Combines momentum and RMSProp with bias-corrected moving averages."),
    ("learning-rate warmup", "Linearly increase LR for early steps to stabilize training."),
    ("mixed-precision training", "Use bf16/fp16 for compute and fp32 master weights for stability."),
    ("gradient checkpointing", "Recompute activations on backward to trade compute for memory."),
    ("PEFT", "Parameter-efficient fine-tuning: train a small fraction of parameters."),
    ("RAG", "Retrieve relevant context from a vector store and condition generation on it."),
    ("FAISS", "Library for efficient similarity search over dense vectors."),
    ("embedding", "Dense vector representation of a token, sentence, or item."),
    ("perplexity", "exp(mean negative log-likelihood); lower is better for LMs."),
    ("ROC AUC", "Area under TPR-vs-FPR curve; threshold-independent classifier quality."),
    ("precision and recall", "Precision = TP/(TP+FP); recall = TP/(TP+FN)."),
    ("F1 score", "Harmonic mean of precision and recall."),
    ("bias-variance tradeoff", "Underfitting (high bias) vs overfitting (high variance)."),
    ("regularization", "Penalize complexity (L1/L2) to improve generalization."),
    ("k-fold cross validation", "Split data into k folds; train on k−1, validate on 1, rotate."),
]

SYSDESIGN_TOPICS = [
    ("load balancer", "Distribute requests across backends; L4 (TCP) or L7 (HTTP) routing."),
    ("rate limiter", "Token-bucket / leaky-bucket to cap request rate per client."),
    ("CDN", "Cache static assets at edge locations close to users to cut latency."),
    ("read-through cache", "Application reads from cache; on miss, fetch from DB and populate."),
    ("write-through cache", "Writes update cache and DB synchronously; reads stay consistent."),
    ("sharding", "Horizontal partitioning of data across nodes by a shard key."),
    ("replication", "Maintain copies of data for HA; sync (strong) or async (lag) modes."),
    ("CAP theorem", "Under partition, choose Consistency or Availability."),
    ("eventual consistency", "Replicas converge over time without immediate sync guarantees."),
    ("idempotency key", "Client-supplied id so retried requests are processed at most once."),
    ("message queue", "Async decoupling; producers enqueue, consumers process at their pace."),
    ("pub/sub", "Multiple subscribers receive messages broadcast on a topic."),
    ("consistent hashing", "Map keys/nodes to a ring so adding nodes moves only ~K/N keys."),
    ("circuit breaker", "Stop calling a failing dependency for a window to let it recover."),
    ("WebSocket", "Full-duplex persistent TCP connection upgraded from HTTP."),
    ("gRPC", "HTTP/2 RPC framework using protobufs for compact, typed messages."),
    ("OAuth2", "Delegated authorization via access tokens issued by an auth server."),
    ("JWT", "Signed token carrying claims; stateless auth via signature verification."),
]

DEBUG_TEMPLATES = [
    ("Python list mutated during iteration",
     "Iterate over a copy (`for x in items[:]`) or build a new list with comprehension; mutating during iteration skips elements."),
    ("KeyError on dict access",
     "Use `dict.get(key, default)` or check membership with `in`; KeyError fires when the key is absent."),
    ("off-by-one in range loop",
     "Recall `range(n)` yields 0..n-1; for inclusive upper bound use `range(n+1)`."),
    ("infinite recursion in DFS",
     "Mark nodes as visited before recursing; without a visited set, cycles cause unbounded recursion."),
    ("CUDA out-of-memory during training",
     "Lower batch size, enable gradient checkpointing, switch to bf16/fp16, or use 4-bit quantization."),
    ("vanishing gradients in deep net",
     "Use residual connections, ReLU/GELU activations, and proper init (He/Xavier)."),
    ("NaN loss after a few steps",
     "Clip gradients, lower learning rate, check for log(0) / division-by-zero, and verify input normalization."),
    ("model overfits training data",
     "Add dropout / weight decay, augment data, use early stopping, or reduce model capacity."),
    ("Git merge conflict",
     "Open the file, resolve the `<<<<<<<` markers manually, then `git add` and `git commit`."),
    ("port already in use",
     "Identify the PID with `lsof -i :PORT` (or `netstat -ano`) and kill it, or pick a different port."),
    ("CORS error in browser",
     "Configure the server to send `Access-Control-Allow-Origin` for the requesting origin."),
    ("slow SQL query",
     "Add an index on the filter columns, avoid SELECT *, and check the EXPLAIN plan."),
]

MATH_TEMPLATES = [
    ("A train travels {a} km in {b} hours. What is its average speed?",
     lambda a, b: f"Average speed = distance / time = {a}/{b} = {a/b:.2f} km/h."),
    ("If a rectangle has length {a} and width {b}, what is its area?",
     lambda a, b: f"Area = length × width = {a} × {b} = {a*b}."),
    ("Compute the sum of integers from 1 to {a}.",
     lambda a, _b: f"Use Gauss's formula: n(n+1)/2 = {a}×{a+1}/2 = {a*(a+1)//2}."),
    ("Find the GCD of {a} and {b}.",
     lambda a, b: f"Apply Euclid's algorithm repeatedly until the remainder is 0; gcd({a},{b}) = {__import__('math').gcd(a,b)}."),
    ("A coin is tossed {a} times. What is the expected number of heads?",
     lambda a, _b: f"E[heads] = n × p = {a} × 0.5 = {a*0.5}."),
    ("Solve for x: {a}x + {b} = {c}.",
     lambda a, b, c=None: f"x = (c − b)/a; with the given values, isolate x by subtracting {b} then dividing by {a}."),
]


# ---------- Generators ----------

def gen_explain(topic: str, summary: str) -> dict:
    phrasings = [
        f"Explain {topic} in 2-3 sentences.",
        f"What is {topic} and when is it useful?",
        f"Briefly describe {topic}.",
        f"Give a concise explanation of {topic}.",
        f"How does {topic} work?",
    ]
    return {
        "instruction": random.choice(phrasings),
        "input": "",
        "output": summary,
    }


def gen_compare(t1, s1, t2, s2) -> dict:
    return {
        "instruction": f"Compare {t1} and {t2}. When would you pick one over the other?",
        "input": "",
        "output": f"{t1}: {s1}\n{t2}: {s2}\nPick {t1} when its properties match the constraints; pick {t2} otherwise.",
    }


def gen_complexity(topic: str, summary: str) -> dict:
    return {
        "instruction": f"What is the time complexity of {topic}, and why?",
        "input": "",
        "output": summary,
    }


def gen_debug(symptom: str, fix: str) -> dict:
    return {
        "instruction": f"My program has this issue: {symptom}. How do I fix it?",
        "input": "",
        "output": fix,
    }


def gen_math() -> dict:
    template, solver = random.choice(MATH_TEMPLATES)
    a = random.randint(2, 200)
    b = random.randint(2, 50)
    c = random.randint(50, 500)
    question = template.format(a=a, b=b, c=c)
    try:
        answer = solver(a, b)
    except TypeError:
        answer = solver(a, b, c)
    return {"instruction": question, "input": "", "output": answer}


def gen_code_snippet(topic: str, summary: str) -> dict:
    examples = {
        "binary search": "def bsearch(a, x):\n    lo, hi = 0, len(a)-1\n    while lo <= hi:\n        m = (lo+hi)//2\n        if a[m] == x: return m\n        if a[m] < x: lo = m+1\n        else: hi = m-1\n    return -1",
        "BFS": "from collections import deque\ndef bfs(g, s):\n    seen, q = {s}, deque([s])\n    while q:\n        u = q.popleft()\n        for v in g[u]:\n            if v not in seen:\n                seen.add(v); q.append(v)\n    return seen",
        "DFS": "def dfs(g, u, seen=None):\n    seen = seen or set()\n    seen.add(u)\n    for v in g[u]:\n        if v not in seen:\n            dfs(g, v, seen)\n    return seen",
        "merge sort": "def msort(a):\n    if len(a) <= 1: return a\n    m = len(a)//2\n    L, R = msort(a[:m]), msort(a[m:])\n    out, i, j = [], 0, 0\n    while i < len(L) and j < len(R):\n        if L[i] <= R[j]: out.append(L[i]); i += 1\n        else: out.append(R[j]); j += 1\n    return out + L[i:] + R[j:]",
    }
    code = examples.get(topic)
    if not code:
        return None
    return {
        "instruction": f"Write a clean Python implementation of {topic}.",
        "input": "",
        "output": f"```python\n{code}\n```\n{summary}",
    }


# ---------- Build ----------

def build_records(n_target: int) -> list[dict]:
    pool: list[dict] = []
    all_topics = [(t, s, "dsa") for t, s in DSA_TOPICS] \
               + [(t, s, "ml") for t, s in ML_TOPICS] \
               + [(t, s, "sys") for t, s in SYSDESIGN_TOPICS]

    while len(pool) < n_target:
        kind = random.choices(
            ["explain", "compare", "complexity", "debug", "math", "code"],
            weights=[35, 15, 15, 15, 10, 10],
        )[0]

        if kind == "explain":
            t, s, _ = random.choice(all_topics)
            pool.append(gen_explain(t, s))
        elif kind == "compare":
            (t1, s1, _), (t2, s2, _) = random.sample(all_topics, 2)
            pool.append(gen_compare(t1, s1, t2, s2))
        elif kind == "complexity":
            t, s, _ = random.choice([x for x in all_topics if x[2] == "dsa"])
            pool.append(gen_complexity(t, s))
        elif kind == "debug":
            sym, fix = random.choice(DEBUG_TEMPLATES)
            pool.append(gen_debug(sym, fix))
        elif kind == "math":
            pool.append(gen_math())
        elif kind == "code":
            t, s, _ = random.choice([x for x in all_topics if x[2] == "dsa"])
            rec = gen_code_snippet(t, s)
            if rec:
                pool.append(rec)

    return pool


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    print("Generating 10,200 records...")
    records = build_records(10_200)
    random.shuffle(records)
    eval_recs = records[:200]
    train_recs = records[200:]
    write_jsonl(TRAIN_PATH, train_recs)
    write_jsonl(EVAL_PATH, eval_recs)
    print(f"  train -> {TRAIN_PATH}  ({len(train_recs)} records)")
    print(f"  eval  -> {EVAL_PATH}  ({len(eval_recs)} records)")


if __name__ == "__main__":
    main()
