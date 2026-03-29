/* Spawn target with __builtin_sync between two segments (barrier ordering). */
typedef unsigned long uint32_t;

uint32_t __builtin_load(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);
void __builtin_sync(void);

uint32_t __builtin_block_idx(void);
uint32_t __builtin_thread_idx(void);
uint32_t __builtin_block_dim(void);
void __builtin_spawn(unsigned int, unsigned int, void (*)(uint32_t), uint32_t);

uint32_t *p_slot = (uint32_t *)6;

void worker(uint32_t base_p)
{
    uint32_t t = __builtin_thread_idx();
    __builtin_sync();
    __builtin_store(base_p + t * 4, t + 1U);
}

void kernel(void)
{
    uint32_t base = __builtin_load(p_slot);
    __builtin_spawn(1, 2, worker, base);
}
