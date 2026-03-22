/* idx = vbroadcast(0) when r == FOREST_HEIGHT (sample wrap-to-root). */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

const uint32_t ROUNDS = 4;
const uint32_t FOREST_HEIGHT = 3;

void kernel() {
    uint32_t idx_p = load(p_inp_indices_p);
    uint32_t val_p = load(p_inp_values_p);
    vec8_t idx = vload(idx_p);
    vec8_t val = vload(val_p);
    for (uint32_t r = 0; r < ROUNDS; r++) {
        idx = idx + vbroadcast(1);
        if (r == FOREST_HEIGHT) {
            idx = vbroadcast(0);
        }
    }
    vstore(idx_p, idx);
    vstore(val_p, val);
}
