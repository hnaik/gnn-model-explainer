"""make_data.py

    A tool for generating graph data
"""
import abc
import json
import sys
import logging

from argparse import ArgumentParser, Namespace
from pathlib import Path
from train import syn_task2
from typing import Callable, List, Tuple

import networkx as nx
import numpy as np

from networkx.readwrite import json_graph

import gengraph

from utils import featgen

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s", level=logging.DEBUG
)
logger = logging.getLogger("make_data")


def make_method(dataset: str) -> Callable:
    return getattr(gengraph, "gen_" + dataset)


def make_graph(dataset: str, input_dim: int) -> Tuple[nx.Graph, List, str]:
    fn = make_method(dataset)
    if dataset == "syn2":
        return fn()
    generator = featgen.ConstFeatureGen(np.ones(input_dim, dtype=float))
    return fn(feature_generator=generator)


def write(graph: nx.Graph, name: str, labels: List, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = output_dir / f"{name}.json"
    for node in graph.nodes:
        graph.nodes[node]["feat"] = graph.nodes[node]["feat"].tolist()

    labels = [int(i) for i in labels]
    graph.graph['labels'] = labels
    graph.graph['name'] = name

    logger.info('Writing to %s', output_name)
    with open(output_name, "w") as f:
        json.dump(json_graph.node_link_data(graph), f)


def main(args: Namespace):
    G, labels, name = make_graph(args.dataset, args.input_dim)
    logger.debug(G.nodes[1])
    logger.debug(f"Generated graph {name} with labels {labels}")
    write(G, name, labels, args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(sys.argv[0])
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["syn1", "syn2", "syn3", "syn4", "syn5"],
        required=True,
    )
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--output-dir", type=Path, default="./output")
    parser.add_argument("--bmname")

    parser.set_defaults(input_dim=10)

    main(parser.parse_args())
