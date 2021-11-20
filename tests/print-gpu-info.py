import cupy
import pprint

if __name__ == '__main__':

    pp = pprint.PrettyPrinter()

    print(f'CUDA driver version is  {cupy.cuda.runtime.driverGetVersion()}\n'
          f'CUDA runtime version is {cupy.cuda.runtime.runtimeGetVersion()}\n')

    for i in range(cupy.cuda.runtime.getDeviceCount()):
        print(f'Properties for device {i}:')
        pp.pprint(cupy.cuda.runtime.getDeviceProperties(i))
