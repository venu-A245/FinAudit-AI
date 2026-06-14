from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DemoConfig:
    data_dir: str = os.getenv("FINAUDIT_DATA_DIR", "data/synthetic")
    model_name: str = os.getenv("FINAUDIT_OLLAMA_MODEL", "llama3.1:8b")
    top_k_per_source: int = int(os.getenv("FINAUDIT_TOP_K", "4"))
    graph_hop_boost: float = float(os.getenv("FINAUDIT_GRAPH_HOP_BOOST", "0.12"))


CONFIG = DemoConfig()
