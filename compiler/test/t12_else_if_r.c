/* else-if chain on scalar r inside unrolled rounds (three-way dispatch). */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);

uint32_t *p_inp_values_p = (uint32_t *)6;

const uint32_t ROUNDS = 3;

void kernel() {
    uint32_t val_p = load(p_inp_values_p);
    vec8_t val = vload(val_p);
    for (uint32_t round = 0; round < ROUNDS; round++) {
        uint32_t r = round % 3;
        vec8_t node;
        if (r == 0) {
            node = vbroadcast(10);
        } else if (r == 1) {
            node = vbroadcast(100);
        } else {
            node = vbroadcast(1000);
        }
        val = val + node;
    }
    vstore(val_p, val);
}
