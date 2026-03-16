# Mohsin Mehmood

**Machine Learning Engineer · LLM Systems & Agent Architectures · Open-Source Contributor**

Forward Deployed ML Engineer at [Kodamai](https://kodamai.com) (Glasgow). I design and build production core ML and AI systems: multi-agent pipelines, document extraction infrastructure, and LLM-powered workflow automation. Most of my work sits at the boundary between production-grade ML and the messy reality of deploying it in regulated, high-stakes environments.

I carefully consider system architecture before writing code, including transaction boundaries, failure modes, inference latency budgets, and how models degrade in production. I have shipped AI systems in healthcare (HIPAA/PDPA), enterprise supply chain, and early-stage startups, working directly with non-technical domain experts to translate complex processes into automated pipelines.

---

## Open-Source Contributions

| Project | Contribution |
|---|---|
| [**google/flax**](https://github.com/google/flax) | **#5 contributor** (last 12 months, official leaderboard · 19 commits). Merged PRs in Flax NNX core: fixed `nnx.tabulate` crash with empty/None values ([#4891](https://github.com/google/flax/pull/4891)); fixed variable hook display bugs in `nnx.tabulate` ([#5008](https://github.com/google/flax/pull/5008)). Both shipped in v0.12.1. |
| [**python/cpython**](https://github.com/python/cpython) | 6 merged PRs: fixed `TextIOWrapper.tell()` assertion failure with standalone carriage return (backported to 3.13 + 3.14 · [#141331](https://github.com/python/cpython/pull/141331)); fixed `re.Match.group()` doc claiming [1..99] range limit ([#144696](https://github.com/python/cpython/pull/144696)); documented asyncio Task cancellation propagation ([#141249](https://github.com/python/cpython/pull/141249)); fixed inaccurate object comparison docs ([#141221](https://github.com/python/cpython/pull/141221)). |
| [**pytorch/pytorch**](https://github.com/pytorch/pytorch) | Approved PR in PyTorch core ([#167209](https://github.com/pytorch/pytorch/pull/167209)). |
| [**uber/causalml**](https://github.com/uber/causalml) | Fixed bug where `estimation_sample_size` was not propagated from `UpliftRandomForestClassifier` to individual trees ([#850](https://github.com/uber/causalml/pull/850)). Silently broken behaviour in the core uplift Cython layer. |

---

## Technical Depth

```python
core_ml = {
    "frameworks":     ["PyTorch", "JAX/Flax", "TensorFlow"],
    "training":       ["LoRA", "QLoRA", "RLHF", "DPO", "mixed precision", "distributed training"],
    "inference":      ["quantization (GPTQ/AWQ/bitsandbytes)", "vLLM", "GGUF/MLX", "TensorRT"],
    "evaluation":     ["RAGAS", "ROUGE", "BERTScore", "bootstrapped CI", "MLflow", "W&B"],
    "architectures":  ["Transformers", "CNNs", "multi-agent systems", "RAG pipelines"],
}

systems = {
    "design":         ["modular monolith", "worker architecture", "state machines", "transactional outbox"],
    "infra":          ["Docker", "Kubernetes", "FastAPI", "Redis", "RabbitMQ", "Celery"],
    "cloud":          ["GCP Vertex AI", "AWS SageMaker / EC2 / Lambda", "Azure ML"],
    "observability":  ["structured logging", "latency tracing", "drift monitoring", "PHI/PII redaction"],
    "compliance":     ["HIPAA", "Singapore PDPA"],
    "languages":      ["Python", "C++", "Bash"],
}
```

---

## Contact

- **Website**: [mohsinmehmood.com](https://mohsinmehmood.com)
- **LinkedIn**: [mohsin-mehmood675](https://linkedin.com/in/mohsin-mehmood675)
- **Location**: Pakistan · Open to relocation
