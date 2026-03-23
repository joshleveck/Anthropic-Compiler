/*
 * Scalar kernel matching `reference_kernel2` in problem.py (lines 552–581).
 *
 * - One scalar idx/val per (batch element), same memory layout as build_mem_image.
 * - Use small compile-time ROUNDS and BATCH_SIZE so the loop unroller fully unrolls
 *   (then `__builtin_debug(...)` gets constant round/batch args — same idea as `__builtin_debug!(...)`).
 *
 * Trace tags for __builtin_debug(value, round, batch, tag):
 *   0 idx  1 val  2 node_val  3 hashed_val  4 next_idx  5 wrapped_idx
 */
typedef unsigned long uint32_t;

uint32_t __builtin_load(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);
uint32_t __builtin_myhash(uint32_t a);
void __builtin_debug(uint32_t value, uint32_t round, uint32_t batch, uint32_t tag);

uint32_t *p_inp_values_p = (uint32_t *)6;
uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_forest_values_p = (uint32_t *)4;
uint32_t *p_n_nodes = (uint32_t *)1;

const uint32_t FOREST_HEIGHT = 10;
const uint32_t ROUNDS = 16;
const uint32_t BATCH_SIZE = 256;

const uint32_t ZERO = 0;
const uint32_t ONE = 1;
const uint32_t TWO = 2;

void kernel()
{
    uint32_t n_nodes = __builtin_load(p_n_nodes);
    uint32_t forest_values_p = __builtin_load(p_forest_values_p);
    uint32_t inp_indices_p = __builtin_load(p_inp_indices_p);
    uint32_t inp_values_p = __builtin_load(p_inp_values_p);

    __builtin_flow_pause();

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        uint32_t idx_addr = inp_indices_p + i;
        uint32_t val_addr = inp_values_p + i;

        uint32_t idx = __builtin_load(idx_addr);
        uint32_t val = __builtin_load(val_addr);

        for (uint32_t h = 0; h < ROUNDS; h++)
        {
            uint32_t round_in_tree = h % (FOREST_HEIGHT + 1);
            uint32_t is_wrap_around_round = round_in_tree == FOREST_HEIGHT;

            __builtin_debug(val, h, i, 1);
            __builtin_debug(idx, h, i, 0);
            uint32_t node_val = __builtin_load(forest_values_p + idx);
            __builtin_debug(node_val, h, i, 2);
            val = __builtin_myhash(val ^ node_val);
            __builtin_debug(val, h, i, 3);
            idx = ((TWO * idx) + ONE) + (val % TWO);
            __builtin_debug(idx, h, i, 4);

            if (is_wrap_around_round)
            {
                idx = ZERO;
            }
            __builtin_debug(idx, h, i, 5);
        }

        __builtin_store(val_addr, val);
        __builtin_store(idx_addr, idx);
        __builtin_sync();
    }

    __builtin_flow_pause();
}
