import torch
import torch.utils.benchmark as benchmark

if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    deviceType = 'cuda'
else:
    deviceType = 'not cuda'

# torch functions
def tensor_det(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    return torch.linalg.det(a)

def tensor_mul(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    b = torch.randn(size, size, device=deviceType)
    return torch.matmul(a, b)

def tensor_inv(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    return torch.linalg.inv(a)

def tensor_LS(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    b = torch.randn(size, 1, device=deviceType)
    return torch.linalg.lstsq(a, b)

def tensor_eig(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    at = torch.t(a)
    sym = torch.add(a, at)
    return torch.linalg.eigh(sym)

def tensor_svd(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    return torch.linalg.svd(a)

def tensor_norm(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    return torch.linalg.norm(a)

def tensor_covar(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    return torch.cov(a)

def tensor_hist(size, deviceType):
    a = torch.randn(size, device=deviceType)
    return torch.histogram(a)

def tensor_einsum(size, deviceType):
    a = torch.randn(size, size, size, device=deviceType)
    b = torch.randn(size, size, size, device=deviceType)
    return torch.einsum('bij,bjk->bik', a, b)

def tensor_mm(size, deviceType):
    a = torch.randn(size, size, device=deviceType)
    b = torch.randn(size, size, device=deviceType)
    return torch.mm(a, b)

def tensor_grad(size, deviceType):
    a = torch.tensor(torch.randn(size, size, size, device=deviceType))
    return torch.gradient(a)
    
# benchmarking function
def benchmark_function(function, deviceType):
    # the size array can be adjusted depending on what the test needs
    size = [10, 100, 1000, 10000]
    
    for num in size:                    
        # determine number of runs for benchmarking test based on function and number of runs
        if function.__name__ == 'tensor_hist' or function.__name__ == 'tensor_norm':
            runs = 100
        else:
            if num >= 10000:
                if function.__name__ == 'tensor_eig' or function.__name__ == 'tensor_svd':
                    runs = 10
                else:
                    runs = 50
            else:
                if num >= 1000:
                    if function.__name__ == 'tensor_einsum' or function.__name__ == 'tensor_grad':
                        runs = 10
                else:
                    runs = 100
        
        # run benchmarking test for function of given size
        timedFunction = benchmark.Timer(
            stmt="function(num, deviceType)",
            globals={"function": function, "num": num, "deviceType": deviceType}
            )
        time = timedFunction.timeit(runs).mean
        print(f"{function.__name__} for matrix/tensor with dimension size of {num} on {deviceType} for {runs} runs    : {time:.3E} s")
        print()

        # prevent function from running the next size if the last benchmarking test took longer than 10 minutes to run
        totalTime = time * runs
        if totalTime >= 600:
            print('Time of last round was greater than or equal to 600 s (10 minutes). Stopping calculations.')
            break

# the function array can be adjusted to include as many or as few of the functions to test
functionList = [tensor_det, tensor_mul, tensor_inv, tensor_LS, tensor_eig, tensor_svd, tensor_norm, tensor_covar, tensor_hist, tensor_einsum, tensor_mm, tensor_grad]

if deviceType == 'cuda':
    for function in functionList:
        print(f"\n{function.__name__} on {torch.cuda.get_device_name()}")
        benchmark_function(function, deviceType)

else:
    # CPU
    torch.set_default_device('cpu')
    deviceType = 'cpu'
    for function in functionList:
        print(f"\n{function.__name__} on {deviceType}")
        benchmark_function(function, deviceType)
        
    # Gaudi
    torch.set_default_device('hpu')
    deviceType = 'hpu'
    for function in functionList:
        print(f"\n{function.__name__} on {deviceType}")
        benchmark_function(function, deviceType)
