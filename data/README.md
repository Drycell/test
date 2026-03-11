# Data

- `mock_connectome/`: deterministic small graph for CI/smoke runs.
- `real_connectome/`: reference C. elegans hermaphrodite edgelist converted to the local CSV schema.

## Real connectome source

`data/real_connectome/*` is derived from:
- OpenWorm c302 dataset: `herm_full_edgelist.csv`
- URL: `https://raw.githubusercontent.com/openworm/c302/master/c302/data/herm_full_edgelist.csv`

Conversion notes:
- `Type=chemical` -> `chemical_synapses.csv`
- `Type=electrical` -> `gap_junctions.csv`
- `neurons.csv` contains all observed neuron IDs with heuristic `is_sensor` and `is_motor` flags based on canonical C. elegans neuron naming prefixes.
