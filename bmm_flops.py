import time
import torch
import numpy as np


def benchmark_bmm(b, m, n, k, num_iterations=100):
    A = torch.randn((b, m, n)).half().to("cuda:0")
    B = torch.randn((b, n, k)).half().to("cuda:0")
    C = torch.empty((b, m, k)).half().to("cuda:0")
    num_warmup_iterations = 50
    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            torch.bmm(A, B, out=C)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    print(f"Elapsed time for {b}x{m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {b}x{m}x{n}x{k}: {(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    flops = (2 * b * m * n * k) / (elapsed_time * 10**12)
    print("-" * 80)
    return flops

def benchmark_bmm_max(b, m, n, k, num_iterations=200):
    A = torch.randn((b, m, n)).half().to("cuda:0")
    B = torch.randn((b, n, k)).half().to("cuda:0")
    C = torch.empty((b, m, k)).half().to("cuda:0")
    num_warmup_iterations=50
    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            torch.bmm(A, B, out=C)
        torch.cuda.synchronize()
        times[i] = time.time()

    #elapsed_time = (time.time() - start_time) / num_iterations
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    elapsed_time = np.amax(times)
    print(f"Elapsed time for {b}x{m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {b}x{m}x{n}x{k}: {(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    flops = (2 * b * m * n * k) / (elapsed_time * 10**12)
    print("-" * 80)
    return flops

def benchmark_bmm_min(b, m, n, k, num_iterations=200):
    A = torch.randn((b, m, n)).half().to("cuda:0")
    B = torch.randn((b, n, k)).half().to("cuda:0")
    C = torch.empty((b, m, k)).half().to("cuda:0")
    num_warmup_iterations=50
    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            torch.bmm(A, B, out=C)
        torch.cuda.synchronize()
        times[i] = time.time()

    #elapsed_time = (time.time() - start_time) / num_iterations
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    elapsed_time = np.amin(times)
    print(f"Elapsed time for {b}x{m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {b}x{m}x{n}x{k}: {(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    flops = (2 * b * m * n * k) / (elapsed_time * 10**12)
    print("-" * 80)
    return flops

def bench_list(b, m, N, k):
    benches = []
    for n in N:
        benches.append(benchmark_bmm(b, m, n, k))
    return benches



if __name__ == '__main__':
    torch.cuda.set_device("cuda:0")

    #shared dimension sweep.
    #N_values= range(64, 2**12, 64)
    #for logb in range(5, 9):
    #    bench_list(b=2**logb, m=2048, N=N_values, k=2048)




    # Try to determine the effect of b on throughput with square individual MMs.
    '''for log_b in range(7):
        b = 2**log_b
        benchmark_bmm(b, m=1024, n=1024, k=1024)
        benchmark_bmm(b, m=2048, n=2048, k=2048)
        benchmark_bmm(b, m=4096, n=4096, k=4096)
        benchmark_bmm(b, m=8192, n=8192, k=8192)
    '''
    # Try to determine the effect of b and outer_dim on throughput with non-square
    # individual MMs.
    for log_b in range(7):
        b = 2**log_b
        for log_outer_dim in range(5, 14):
            outer_dim = 2**log_outer_dim
            benchmark_bmm_min(b, m=outer_dim, n=4096, k=outer_dim)
    '''
    h = 2048
    m = 2048
    k = int(h)
    n = h
    b = 512

    A = torch.randn((b, m, n)).half().to("cuda:0")
    B = torch.randn((b, n, k)).half().to("cuda:0")
    C = torch.empty((b, m, k)).half().to("cuda:0")
    torch.bmm(A, B, out=C)
    '''