typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t __builtin_load(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);

uint32_t __builtin_block_idx(void);
uint32_t __builtin_thread_idx(void);
uint32_t __builtin_block_dim(void);
void __builtin_spawn(unsigned int, unsigned int,
                     void (*)(uint32_t),
                     uint32_t);

uint32_t *p_inp_values_p = (uint32_t *)6;

void worker(uint32_t base_p)
{
    uint32_t i = __builtin_block_idx() * __builtin_block_dim() + __builtin_thread_idx();
    uint32_t addr = base_p + i * 8;
    vec8_t a = __builtin_vbroadcast(i);
    __builtin_vstore(addr, a);
}

void kernel()
{
    uint32_t p = __builtin_load(p_inp_values_p);
    __builtin_spawn(1, 8, worker, p);
}
