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
typedef uint32_t vec8_t[8];

uint32_t __builtin_load(uint32_t addr);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vhash(vec8_t a);
void __builtin_debug(uint32_t value, uint32_t round, uint32_t batch, uint32_t tag);
vec8_t __builtin_vbroadcast(uint32_t s);

uint32_t *p_inp_values_p = (uint32_t *)6;
uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_forest_values_p = (uint32_t *)4;
uint32_t *p_n_nodes = (uint32_t *)1;

const uint32_t FOREST_HEIGHT = 10;
const uint32_t ROUNDS = 16;
const uint32_t BATCH_SIZE = 256;
const uint32_t VLEN = 8;

const uint32_t ZERO = 0;
const uint32_t ONE = 1;
const uint32_t TWO = 2;

void kernel()
{
    uint32_t n_nodes = __builtin_load(p_n_nodes);
    uint32_t forest_values_p = __builtin_load(p_forest_values_p);
    uint32_t inp_indices_p = __builtin_load(p_inp_indices_p);
    uint32_t inp_values_p = __builtin_load(p_inp_values_p);

    const vec8_t VZERO = __builtin_vbroadcast(ZERO);
    const vec8_t VONE = __builtin_vbroadcast(ONE);
    const vec8_t VTWO = __builtin_vbroadcast(TWO);
    const vec8_t VFOREST_VALUES_P = __builtin_vbroadcast(forest_values_p);

    const uint32_t FOREST_ZERO_VALUE = __builtin_load(forest_values_p);

    __builtin_flow_pause(); // needed for reference kernel

    for (uint32_t i = 0; i < BATCH_SIZE; i += VLEN)
    {
        uint32_t idx_addr = inp_indices_p + i;
        uint32_t val_addr = inp_values_p + i;

        vec8_t idx = __builtin_vload(idx_addr);
        vec8_t val = __builtin_vload(val_addr);

        for (uint32_t h = 0; h < ROUNDS; h++)
        {
            uint32_t round_in_tree = h % (FOREST_HEIGHT + 1);
            uint32_t is_wrap_around_round = round_in_tree == FOREST_HEIGHT;
            uint32_t is_zero_round = round_in_tree == 0;

            if (is_zero_round)
            {
                node_val = __builtin_vbroadcast(FOREST_ZERO_VALUE);
            }
            else
            {
                vec8_t forest_indicies = VFOREST_VALUES_P + idx;
                vec8_t node_val;
                for (int vi = 0; vi < VLEN; vi++)
                {
                    node_val[vi] = __builtin_load(forest_indicies[vi]);
                }
            }

            // __builtin_debug(node_val, h, i, 2);
            val = __builtin_vhash(val ^ node_val);
            // __builtin_debug(val, h, i, 3);
            vec8_t val_mod_two = val % VTWO;
            vec8_t two_idx_plus_one = VTWO * idx + VONE;
            idx = two_idx_plus_one + val_mod_two;
            // __builtin_debug(idx, h, i, 4);

            if (is_wrap_around_round)
            {
                idx = VZERO;
            }
            // __builtin_debug(idx, h, i, 5);
        }

        __builtin_vstore(val_addr, val);
        __builtin_vstore(idx_addr, idx);
        __builtin_sync();
    }

    __builtin_flow_pause();
}
