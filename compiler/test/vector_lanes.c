/*
 * Vector-lane kernel using __builtin_spawn(blocks, threads, execute_rounds, ...).
 * Each block must advance the batch pointer by VLEN words (same as `for (i += VLEN)` in sample_vector_reference.c).
 * Example: __builtin_spawn(32, 1, ...) matches BATCH_SIZE 256 with one vec8 chunk per block.
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
void __builtin_flow_pause(void);

uint32_t __builtin_block_idx(void);
uint32_t __builtin_thread_idx(void);
uint32_t __builtin_block_dim(void);
void __builtin_spawn(unsigned int, unsigned int,
                     void (*)(uint32_t, uint32_t, uint32_t, uint32_t, vec8_t, vec8_t, vec8_t, vec8_t),
                     uint32_t, uint32_t, uint32_t, uint32_t, vec8_t, vec8_t, vec8_t, vec8_t);

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

void execute_rounds(uint32_t inp_indices_p, uint32_t inp_values_p, uint32_t forest_values_p,
                    uint32_t forest_zero_value, vec8_t vforest_values_p, vec8_t vzero,
                    vec8_t vone, vec8_t vtwo, vec8_t vforest_one_value, vec8_t vforest_two_value)
{
    uint32_t i = (__builtin_block_dim() * __builtin_block_idx() + __builtin_thread_idx()) * VLEN;
    uint32_t idx_addr = inp_indices_p + i;
    uint32_t val_addr = inp_values_p + i;

    vec8_t idx = __builtin_vload(idx_addr);
    vec8_t val = __builtin_vload(val_addr);

    vec8_t node_val;

    for (uint32_t h = 0; h < ROUNDS; h++)
    {
        uint32_t round_in_tree = h % (FOREST_HEIGHT + 1);
        uint32_t is_wrap_around_round = round_in_tree == FOREST_HEIGHT;
        uint32_t is_zero_round = round_in_tree == 0;
        uint32_t is_one_two_round = round_in_tree == 1;

        if (is_zero_round)
        {
            node_val = __builtin_vbroadcast(forest_zero_value);
        }
        else if (is_one_two_round)
        {
            uint32_t idx_minus_one = idx - vone;
            node_val = __builtin_vselect(idx_minus_one, vforest_two_value, vforest_one_value);
        }
        else
        {
            vec8_t forest_indicies = vforest_values_p + idx;
            for (int vi = 0; vi < (int)VLEN; vi++)
            {
                node_val[vi] = __builtin_load(forest_indicies[vi]);
            }
        }

        val = __builtin_vhash(val ^ node_val);
        uint32_t val_bit = val & vone;
        uint32_t two_idx_plus_one = idx + idx + vone;
        idx = two_idx_plus_one + val_bit;

        if (is_wrap_around_round)
        {
            idx = vzero;
        }
        __builtin_sync();
    }

    __builtin_vstore(val_addr, val);
    __builtin_vstore(idx_addr, idx);
    __builtin_sync();
}

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
    const uint32_t FOREST_ONE_VALUE = __builtin_load(forest_values_p + 1);
    const vec8_t VFOREST_ONE_VALUE = __builtin_vbroadcast(FOREST_ONE_VALUE);
    const uint32_t FOREST_TWO_VALUE = __builtin_load(forest_values_p + 2);
    const vec8_t VFOREST_TWO_VALUE = __builtin_vbroadcast(FOREST_TWO_VALUE);

    __builtin_flow_pause();

    __builtin_spawn(1, 32, execute_rounds, inp_indices_p, inp_values_p, forest_values_p, FOREST_ZERO_VALUE,
                    VFOREST_VALUES_P, VZERO, VONE, VTWO, VFOREST_ONE_VALUE, VFOREST_TWO_VALUE);

    __builtin_flow_pause();
}
