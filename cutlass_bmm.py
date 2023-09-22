import cutlass
import torch
import random
import time

dtype = torch.float16
plan = cutlass.op.Gemm(element=dtype, layout=cutlass.LayoutType.RowMajor)

file = "results/cutlass_test_in_place.txt"


def initialize(dtype, M, N, K, b):
    torch.cuda.empty_cache()
    sizes = [(b, M, K), ( K, N), (b, M, N)]
    return [torch.randint(-3, 3, size, device='cuda').to(dtype) for size in sizes]


for h in range(2**13,2**15+64,2*64): #range(20608-128, 22272+128+64, 64):
    tiles = plan.tile_descriptions()
    
    '''if h >= 20608 and h <= 22272:
        td = tiles[0] #tiles[0] is 256x128
    else:
        td = tiles[1] #tiles[1] is 128x256'''
    #plan.compile(td)
    m = 4# int(32)
    k = int(4*h)
    n = h
    b=2048
    As, Bs, Cs = initialize(dtype, m, n, k, b)
    torch.cuda.empty_cache()
    plan.compile()
    #print(torch.cuda.memory_summary(device='cuda'))
    num_warm = 50
    num_iterations=200
    for i in range( num_warm + num_iterations):
        if i == num_warm:
            start_time = time.time()
        plan.run(As, Bs, Cs, Cs, sync=True)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    del As, Bs, Cs
    torch.cuda.empty_cache()
    with open(file, 'a') as f:
        f.write(f"Elapsed time for {m}x{n}x{k}, b={b}: {elapsed_time:.4f}\n")
        f.write(f"Throughput (in TFLOP/s) for {m}x{n}x{k}, b={b}: "
            f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}\n") 
        f.write("-" * 80)
        f.write("\n")
'''
tiles = plan.tile_descriptions()
td = tiles[0]
for m in [512,1024]:
    for k in [512,1024,2048]:
        for n in [2048]:
            for b in [128,256,512]:
                print(f"m: {m}, k: {k}, n: {n}, b: {b}")
                plan.compile(td)
                As, Bs, Cs, Ds, = initialize(dtype, m, n, k, b)
                num_warm = 20
                for i in range( num_warm + 20):
                    if i == num_warm:
                        start_time = time.time()
                    plan.run(As, Bs, Cs, Ds, sync=True)
                    torch.cuda.synchronize()
                elapsed_time = (time.time() - start_time) / 20
                with open(file, 'a') as f:
                    f.write(f"Elapsed time for {m}x{n}x{k}, b={b}: {elapsed_time:.4f}\n")
                    f.write(f"Throughput (in TFLOP/s) for {m}x{n}x{k}, b={b}: "
                        f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}\n") 
                    f.write("-" * 80)
                    f.write("\n")
'''
'''
for i in range(100):
    td = tiles[i]
    plan.compile(td)
    As, Bs, Cs, Ds, = generate_problems(b, m, n, k)
    plan.run(As, Bs, Cs, Ds, print_module=True)
    num_warm = 10
    for i in range( num_warm + 20):
        if i == num_warm:
            start_time = time.time()
        plan.run(As, Bs, Cs, Ds, print_module=True)
    elapsed_time = (time.time() - start_time) / 20
    with open(file, 'a') as f:
        f.write(f"Tile description: \n {td} \n")
        f.write(f"Elapsed time for {m}x{n}x{k}, b={b}: {elapsed_time:.4f}\n")
        f.write(f"Throughput (in TFLOP/s) for {m}x{n}x{k}, b={b}: "
            f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}\n") 
        f.write("-" * 80)
        f.write("\n")
'''

