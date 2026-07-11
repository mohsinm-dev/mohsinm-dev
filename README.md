# Mohsin Mehmood

**Software Engineer · AI/ML Systems, Backend & Distributed Platforms**

I build production ML/LLM and agent systems in Python and Julia, and I fix the tools underneath them — with merged contributions to the internals of Flax NNX and the CPython interpreter. Most of my work sits at the boundary between production-grade ML and the messy reality of deploying it in regulated, high-stakes environments: healthcare (Singapore PDPA), enterprise supply chain, and early-stage startups.

---

## Open-Source Contributions

| Project | Contribution |
|---|---|
| [**google/flax**](https://github.com/google/flax) | **#4 contributor by commits over the trailing 12 months** ([contributor graph](https://github.com/google/flax/graphs/contributors)) — 23 commits, +1,165 / −524, all merged into the **NNX core**. 10 merged PRs fixing module-system, transform (`jit`/`vmap`), and dtype-promotion bugs, including [#4891](https://github.com/google/flax/pull/4891) and [#5008](https://github.com/google/flax/pull/5008), both shipped in **v0.12.1**. |
| [**python/cpython**](https://github.com/python/cpython) | 5 merged fixes. Fixed a hard crash in the C text-I/O core — `TextIOWrapper.tell()` assertion failure on standalone carriage return ([#141331](https://github.com/python/cpython/pull/141331)), backported to **Python 3.13 and 3.14**. Plus four merged documentation fixes: `re.Match.group()` range claim ([#144696](https://github.com/python/cpython/pull/144696)), asyncio Task cancellation propagation ([#141249](https://github.com/python/cpython/pull/141249)), `BufferedRandom` inheritance ([#141629](https://github.com/python/cpython/pull/141629)), and object comparison semantics ([#141221](https://github.com/python/cpython/pull/141221)). |
| [**pytorch/pytorch**](https://github.com/pytorch/pytorch) | Approved core PR ([#167209](https://github.com/pytorch/pytorch/pull/167209)). |
| [**uber/causalml**](https://github.com/uber/causalml) | Fixed `estimation_sample_size` not propagating from `UpliftRandomForestClassifier` to individual trees ([#850](https://github.com/uber/causalml/pull/850)) — silently broken behaviour in the core uplift Cython layer. |

---

## Stack

```python
languages = ["Python", "Julia", "C/C++ (CPython core, Cython)", "SQL", "Bash"]

genai_and_agents = [
    "LLM agent & tool-use architectures", "LangChain", "LangGraph", "PydanticAI",
    "MCP servers", "RAG (hybrid dense+sparse retrieval, reranking)",
    "multi-provider model routing & failover", "structured outputs",
    "LoRA / SFT fine-tuning",
]

ml_and_eval = [
    "PyTorch", "JAX/Flax (NNX internals)", "XGBoost", "scikit-learn",
    "HuggingFace", "DeepEval", "LLM-as-judge", "regression suites",
]

backend_and_infra = [
    "async/concurrent pipelines", "microservices (REST + message queues)",
    "blue/green deploys", "distributed tracing (OpenTelemetry)",
    "GCP (Vertex AI, BigQuery)", "AWS (EC2, Lambda, S3)",
    "Docker", "GitHub Actions CI/CD",
    "PostgreSQL", "Redis", "Qdrant", "RabbitMQ", "Airflow",
]
```

---

## Contact

- **Website**: [mohsinmehmood.com](https://mohsinmehmood.com)
- **LinkedIn**: [mohsinmehmood-m](https://www.linkedin.com/in/mohsinmehmood-m/)
- **Email**: to.mohsinmehmood@gmail.com
- **Location**: Pakistan · Open to relocation
