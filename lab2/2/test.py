num_process = [1,4,16]
n = [128, 256, 512, 1024, 2048]
import subprocess
subprocess.run("make build", shell=True)
print('-'*8)
for i in range(len(num_process)):
    for j in range(len(n)):
        cmd = "mpirun -np "+str(num_process[i])+" ./main "+f'{str(n[j])} '
        print('\x1b[32m', cmd, '\x1b[0m')
        subprocess.run(cmd, shell=True)