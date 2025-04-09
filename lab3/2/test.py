num_process = [1,2,4,8,16]
n = [1000000, 32000000, 64000000, 96000000, 128000000]
import subprocess
subprocess.run("make", shell=True)
print('-'*8)
for i in range(len(num_process)):
    for j in range(len(n)):
        cmd = "./parallel_sum "+str(n[j])+" "+str(num_process[i])
        print('\x1b[32m', cmd, '\x1b[0m')
        subprocess.run(cmd, shell=True)