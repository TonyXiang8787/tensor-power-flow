import numpy as np
import power_grid_model as pgm

# standard u rated
u_rated = 10e3
frequency = 50.0

# source
source_sk = 1e20
source_rx = 0.1
source_01 = 1.0
source_u_ref = 1.0
source_node = 0

# cable parameter per km
# 630Al XLPE 10 kV with neutral conductor
cable_type = "630Al"
cable_param = {
    "r1": 0.063,
    "x1": 0.103,
    "c1": 0.0,
    "c0": 0.0,
    "tan1": 0.0,
    "tan0": 0.0,
    "i_n": np.nan,
}
cable_param_pp = {
    "c_nf_per_km": cable_param["c1"] * 1e9,
    "r_ohm_per_km": cable_param["r1"],
    "x_ohm_per_km": cable_param["x1"],
    "g_us_per_km": cable_param["tan1"] * cable_param["c1"] * 2 * np.pi * frequency * 1e6,
    "c0_nf_per_km": cable_param["c0"] * 1e9,
    "g0_us_per_km": cable_param["tan0"] * cable_param["c0"] * 2 * np.pi * frequency * 1e6,
    "max_i_ka": cable_param["i_n"] * 1e-3,
}


def generate_fictional_grid(
    n_feeder: int,
    n_node_per_feeder: int,
    cable_length_km_min: float,
    cable_length_km_max: float,
    load_p_w_max: float,
    load_p_w_min: float,
    pf: float,
    n_step: int,
    load_scaling_min: float,
    load_scaling_max: float,
    seed=0,
):
    rng = np.random.default_rng(seed)

    n_node = n_feeder * n_node_per_feeder + 1
    pgm_dataset = dict()

    # node
    # pgm
    pgm_dataset["node"] = pgm.initialize_array("input", "node", n_node)
    pgm_dataset["node"]["id"] = np.arange(n_node, dtype=np.int32)
    pgm_dataset["node"]["u_rated"] = u_rated

    # line
    n_line = n_node - 1
    to_node_feeder = np.arange(1, n_node_per_feeder + 1, dtype=np.int32)
    to_node_feeder = to_node_feeder.reshape(1, -1) + np.arange(0, n_feeder).reshape(-1, 1) * n_node_per_feeder
    to_node = to_node_feeder.ravel()
    from_node_feeder = np.arange(1, n_node_per_feeder, dtype=np.int32)
    from_node_feeder = from_node_feeder.reshape(1, -1) + np.arange(0, n_feeder).reshape(-1, 1) * n_node_per_feeder
    from_node_feeder = np.concatenate((np.zeros(shape=(n_feeder, 1), dtype=np.int32), from_node_feeder), axis=1)
    from_node = from_node_feeder.ravel()
    length = rng.uniform(low=cable_length_km_min, high=cable_length_km_max, size=n_line)
    # pgm
    pgm_dataset["line"] = pgm.initialize_array("input", "line", n_line)
    pgm_dataset["line"]["id"] = np.arange(n_node, n_node + n_line, dtype=np.int32)
    pgm_dataset["line"]["from_node"] = from_node
    pgm_dataset["line"]["to_node"] = to_node
    pgm_dataset["line"]["from_status"] = 1
    pgm_dataset["line"]["to_status"] = 1
    for attr_name, attr in cable_param.items():
        if attr_name in ["i_n", "tan1", "tan0"]:
            pgm_dataset["line"][attr_name] = attr
        else:
            pgm_dataset["line"][attr_name] = attr * length

    # add load
    n_load = n_node - 1
    # pgm
    pgm_dataset["sym_load"] = pgm.initialize_array("input", "sym_load", n_load)
    pgm_dataset["sym_load"]["id"] = np.arange(n_node + n_line, n_node + n_line + n_load, dtype=np.int32)
    pgm_dataset["sym_load"]["node"] = pgm_dataset["node"]["id"][1:]
    pgm_dataset["sym_load"]["status"] = 1
    pgm_dataset["sym_load"]["type"] = pgm.LoadGenType.const_power
    pgm_dataset["sym_load"]["p_specified"] = rng.uniform(low=load_p_w_min / 3.0, high=load_p_w_max / 3.0, size=n_load)
    pgm_dataset["sym_load"]["q_specified"] = pgm_dataset["sym_load"]["p_specified"] * np.sqrt(1 - pf**2) / pf

    # source
    # pgm
    source_id = n_node + n_line + n_load
    pgm_dataset["source"] = pgm.initialize_array("input", "source", 1)
    pgm_dataset["source"]["id"] = source_id
    pgm_dataset["source"]["node"] = source_node
    pgm_dataset["source"]["status"] = 1
    pgm_dataset["source"]["u_ref"] = source_u_ref
    pgm_dataset["source"]["sk"] = source_sk
    pgm_dataset["source"]["rx_ratio"] = source_rx
    pgm_dataset["source"]["z01_ratio"] = source_01

    # generate time series
    rng = np.random.default_rng(seed)

    # pgm
    n_load = pgm_dataset["sym_load"].size
    scaling = rng.uniform(low=load_scaling_min, high=load_scaling_max, size=(n_step, n_load))
    sym_load_profile = pgm.initialize_array("update", "sym_load", (n_step, n_load))
    sym_load_profile["id"] = pgm_dataset["sym_load"]["id"].reshape(1, -1)
    sym_load_profile["p_specified"] = pgm_dataset["sym_load"]["p_specified"].reshape(1, -1) * scaling
    sym_load_profile["q_specified"] = pgm_dataset["sym_load"]["q_specified"].reshape(1, -1) * scaling

    return {
        "pgm_dataset": pgm_dataset,
        "pgm_update_dataset": {"sym_load": sym_load_profile},
    }
