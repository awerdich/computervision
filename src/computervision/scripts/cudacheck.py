from computervision.datasets import get_gpu_info
device, device_str = get_gpu_info()
print(f'Current device {device}')
print(f'Current device string {device_str}')
