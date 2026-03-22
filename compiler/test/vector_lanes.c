/*
 * Vector-lane kernel using spawn(blocks, threads, execute_rounds, ...).
 * Replace spawn(2, 2, ...) with spawn(128, 2, ...) for full batch coverage (larger compile).
 */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void store(uint32_t addr, uint32_t v);
void vstore(uint32_t addr, vec8_t v);
vec8_t vhash(vec8_t a);
void debug(uint32_t value, uint32_t round, uint32_t batch, uint32_t tag);
vec8_t vbroadcast(uint32_t s);
void flow_pause(void);

uint32_t __builtin_block_idx(void);
uint32_t __builtin_thread_idx(void);
uint32_t __builtin_block_dim(void);
void spawn(unsigned int, unsigned int,
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
                    vec8_t vone, vec8_t vtwo) {
    uint32_t i = __builtin_block_idx() * __builtin_block_dim() + __builtin_thread_idx();
    uint32_t idx_addr = inp_indices_p + i;
    uint32_t val_addr = inp_values_p + i;

    vec8_t idx = vload(idx_addr);
    vec8_t val = vload(val_addr);

    vec8_t node_val;

    for (uint32_t h = 0; h < ROUNDS; h++) {
        uint32_t round_in_tree = h % (FOREST_HEIGHT + 1);
        uint32_t is_wrap_around_round = round_in_tree == FOREST_HEIGHT;
        uint32_t is_zero_round = round_in_tree == 0;

        if (is_zero_round) {
            node_val = vbroadcast(forest_zero_value);
        } else {
            vec8_t forest_indicies = vforest_values_p + idx;
            for (int vi = 0; vi < (int)VLEN; vi++) {
                node_val[vi] = load(forest_indicies[vi]);
            }
        }

        val = vhash(val ^ node_val);
        vec8_t val_mod_two = val % vtwo;
        vec8_t two_idx_plus_one = vtwo * idx + vone;
        idx = two_idx_plus_one + val_mod_two;

        if (is_wrap_around_round) {
            idx = vzero;
        }
    }

    vstore(val_addr, val);
    vstore(idx_addr, idx);
}

void kernel() {
    uint32_t n_nodes = load(p_n_nodes);
    uint32_t forest_values_p = load(p_forest_values_p);
    uint32_t inp_indices_p = load(p_inp_indices_p);
    uint32_t inp_values_p = load(p_inp_values_p);

    const vec8_t VZERO = vbroadcast(ZERO);
    const vec8_t VONE = vbroadcast(ONE);
    const vec8_t VTWO = vbroadcast(TWO);
    const vec8_t VFOREST_VALUES_P = vbroadcast(forest_values_p);

    const uint32_t FOREST_ZERO_VALUE = load(forest_values_p);

    flow_pause();

    spawn(2, 2, execute_rounds, inp_indices_p, inp_values_p, forest_values_p, FOREST_ZERO_VALUE,
          VFOREST_VALUES_P, VZERO, VONE, VTWO);

    flow_pause();
}
