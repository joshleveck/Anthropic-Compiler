typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t load(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);

uint32_t __builtin_block_idx(void);
uint32_t __builtin_thread_idx(void);
uint32_t __builtin_block_dim(void);
void spawn(unsigned int, unsigned int,
           void (*)(uint32_t),
           uint32_t);

uint32_t *p_inp_values_p = (uint32_t *)6;

void worker(uint32_t base_p) {
    uint32_t i = __builtin_block_idx() * __builtin_block_dim() + __builtin_thread_idx();
    uint32_t addr = base_p + i * 8;
    vec8_t a = vbroadcast(i);
    vstore(addr, a);
}

void kernel() {
    uint32_t p = load(p_inp_values_p);
    spawn(2, 2, worker, p);
}
