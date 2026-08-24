"""F5 — Camada de proveniência (W3C PROV-O).

Diferente do beta, o grafo aqui **não é um mockup**: ele é montado a partir da
execução que acabou de rodar na tela (hashes, parâmetros e valores reais), e
pode ser exportado como Cypher (Neo4j) ou PROV-JSON.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .risk import RiskResult

PROV_COLORS = {
    "Entity": "#38BDF8",
    "Activity": "#F59E0B",
    "Agent": "#EC4899",
    "Result": "#10B981",
}


@dataclass
class ProvGraph:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    def add_node(self, node_id: str, kind: str, label: str, **props) -> str:
        if not any(n["id"] == node_id for n in self.nodes):
            self.nodes.append({"id": node_id, "kind": kind, "label": label, "props": props})
        return node_id

    def add_edge(self, source: str, target: str, relation: str) -> None:
        edge = {"source": source, "target": target, "relation": relation}
        if edge not in self.edges:
            self.edges.append(edge)


def environment_fingerprint() -> dict:
    """Ambiente de execução — parte do PROV Agent (reprodutibilidade)."""
    import numpy
    import pandas
    import scipy

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
    }
    try:
        import arch

        versions["arch"] = arch.__version__
    except Exception:
        versions["arch"] = "indisponível"
    versions["executable"] = sys.executable
    return versions


def build_graph(
    ticker: str,
    source: str,
    period: str,
    n_prices: int,
    returns_hash: str,
    results: list[RiskResult],
) -> ProvGraph:
    """Monta o grafo PROV de uma execução real do motor de risco."""
    g = ProvGraph()
    env = environment_fingerprint()
    agent_id = "agent:finprov-engine"

    g.add_node(
        agent_id, "Agent", f"Agent\nfinprov-engine\npy {env['python']}",
        software="finprov_demo", **env,
    )
    raw_id = f"entity:prices:{ticker}"
    g.add_node(
        raw_id, "Entity", f"Entity\nPreços {ticker}",
        ticker=ticker, source=source, period=period, n_obs=n_prices,
    )
    etl_id = f"activity:etl:{ticker}"
    g.add_node(etl_id, "Activity", "Activity\nETL · log-retornos",
               transform="diff(log(close))", n_out=n_prices - 1)
    ret_id = f"entity:returns:{returns_hash}"
    g.add_node(ret_id, "Entity", f"Entity\nRetornos\nsha256:{returns_hash[:8]}…",
               input_hash=returns_hash, n_obs=n_prices - 1)

    g.add_edge(etl_id, raw_id, "used")
    g.add_edge(ret_id, etl_id, "wasGeneratedBy")
    g.add_edge(ret_id, raw_id, "wasDerivedFrom")
    g.add_edge(etl_id, agent_id, "wasAssociatedWith")

    for r in results:
        act_id = f"activity:{r.method}:{r.run_id[:8]}"
        res_id = f"entity:result:{r.run_id[:8]}"
        g.add_node(
            act_id, "Activity", f"Activity\n{r.method}()",
            method=r.method, confidence_level=r.confidence_level,
            horizon_days=r.horizon_days, run_id=r.run_id,
            **{k: v for k, v in r.params.items() if not isinstance(v, (dict, list))},
        )
        g.add_node(
            res_id, "Result", f"{r.method}\n{r.value:.2%}",
            value=r.value, method=r.method, confidence_level=r.confidence_level,
            computed_at=r.computed_at, run_id=r.run_id,
        )
        g.add_edge(act_id, ret_id, "used")
        g.add_edge(res_id, act_id, "wasGeneratedBy")
        g.add_edge(act_id, agent_id, "wasAssociatedWith")
        g.add_edge(res_id, ret_id, "wasDerivedFrom")
    return g


def _cypher_props(props: dict) -> str:
    parts = []
    for k, v in props.items():
        if v is None:
            continue
        key = k.replace("-", "_").replace(" ", "_")
        if isinstance(v, bool):
            parts.append(f"{key}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{key}: {v}")
        else:
            parts.append(f'{key}: "{str(v)}"')
    return ", ".join(parts)


def to_cypher(graph: ProvGraph) -> str:
    """Script Cypher idempotente (MERGE) — cola direto no Neo4j Browser."""
    lines = ["// FinProv — grafo de proveniência (W3C PROV-O)",
             f"// gerado em {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    for n in graph.nodes:
        label = "Result" if n["kind"] == "Result" else n["kind"]
        props = {"id": n["id"], **n["props"]}
        lines.append(f"MERGE (n:{label} {{id: \"{n['id']}\"}})\n  SET n += {{{_cypher_props(props)}}};")
    lines.append("")
    for e in graph.edges:
        lines.append(
            f'MATCH (a {{id: "{e["source"]}"}}), (b {{id: "{e["target"]}"}})\n'
            f'  MERGE (a)-[:{e["relation"]}]->(b);'
        )
    return "\n".join(lines)


def to_prov_json(graph: ProvGraph) -> str:
    """Serialização PROV-JSON (W3C) — o formato canônico de intercâmbio."""
    doc: dict = {
        "prefix": {"finprov": "https://finprov.ufrj.br/prov#",
                   "prov": "http://www.w3.org/ns/prov#"},
        "entity": {}, "activity": {}, "agent": {},
        "used": {}, "wasGeneratedBy": {}, "wasAssociatedWith": {}, "wasDerivedFrom": {},
    }
    for n in graph.nodes:
        bucket = {"Entity": "entity", "Result": "entity",
                  "Activity": "activity", "Agent": "agent"}[n["kind"]]
        doc[bucket][f"finprov:{n['id']}"] = {
            "prov:label": n["label"].replace("\n", " · "),
            **{f"finprov:{k}": v for k, v in n["props"].items()},
        }
    counters: dict[str, int] = {}
    role = {"used": ("prov:activity", "prov:entity"),
            "wasGeneratedBy": ("prov:entity", "prov:activity"),
            "wasAssociatedWith": ("prov:activity", "prov:agent"),
            "wasDerivedFrom": ("prov:generatedEntity", "prov:usedEntity")}
    for e in graph.edges:
        rel = e["relation"]
        counters[rel] = counters.get(rel, 0) + 1
        a, b = role[rel]
        doc[rel][f"finprov:{rel}{counters[rel]}"] = {
            a: f"finprov:{e['source']}", b: f"finprov:{e['target']}"
        }
    return json.dumps(doc, indent=2, ensure_ascii=False, default=str)
