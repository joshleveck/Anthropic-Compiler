/*
 * Scalar kernel matching `reference_kernel2` in problem.py (lines 552–581).
 *
 * - One scalar idx/val per (batch element), same memory layout as build_mem_image.
 * - Use small compile-time ROUNDS and BATCH_SIZE so the loop unroller fully unrolls
 *   (then `debug(...)` gets constant round/batch args — same idea as `debug!(...)`).
 *
 * Trace tags for debug(value, round, batch, tag):
 *   0 idx  1 val  2 node_val  3 hashed_val  4 next_idx  5 wrapped_idx
 */
typedef unsigned long uint32_t;

uint32_t load(uint32_t addr);
void store(uint32_t addr, uint32_t v);
uint32_t myhash(uint32_t a);
void debug(uint32_t value, uint32_t round, uint32_t batch, uint32_t tag);

uint32_t *p_inp_values_p = (uint32_t *)6;
uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_forest_values_p = (uint32_t *)4;
uint32_t *p_n_nodes = (uint32_t *)1;

const uint32_t ROUNDS = 16;
const uint32_t BATCH_SIZE = 256;

void kernel()
{
    uint32_t n_nodes = load(p_n_nodes);
    uint32_t forest_values_p = load(p_forest_values_p);
    uint32_t inp_indices_p = load(p_inp_indices_p);
    uint32_t inp_values_p = load(p_inp_values_p);

    flow_pause();

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        uint32_t idx = load(inp_indices_p + i);
        uint32_t val = load(inp_values_p + i);
        for (uint32_t h = 0; h < ROUNDS; h++)
        {
            debug(val, h, i, 1);
            debug(idx, h, i, 0);
            uint32_t node_val = load(forest_values_p + idx);
            debug(node_val, h, i, 2);
            val = myhash(val ^ node_val);
            debug(val, h, i, 3);
            uint32_t inc;
            if (val % 2 == 0)
            {
                inc = 1;
            }
            else
            {
                inc = 2;
            }
            uint32_t next_idx = 2 * idx + inc;
            debug(next_idx, h, i, 4);

            if (next_idx < n_nodes)
            {
                idx = next_idx;
            }
            else
            {
                idx = 0;
            }
            debug(idx, h, i, 5);
        }

        store(inp_values_p + i, val);
        store(inp_indices_p + i, idx);
    }

    flow_pause();
}
