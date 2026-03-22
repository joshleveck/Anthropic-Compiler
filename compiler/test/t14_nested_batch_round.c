/* Nested loops: batch stride + inner round loop (structure like sample.c). */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

const uint32_t BATCH_SIZE = 8;
const uint32_t ROUNDS = 2;

void kernel() {
    uint32_t inp_indices_p = load(p_inp_indices_p);
    uint32_t inp_values_p = load(p_inp_values_p);
    for (uint32_t batch = 0; batch < BATCH_SIZE; batch += 8) {
        uint32_t idx_addr = inp_indices_p + batch;
        uint32_t val_addr = inp_values_p + batch;
        vec8_t idx = vload(idx_addr);
        vec8_t val = vload(val_addr);
        for (uint32_t round = 0; round < ROUNDS; round++) {
            val = val + vbroadcast(3);
        }
        vstore(idx_addr, idx);
        vstore(val_addr, val);
    }
}
