num_process = [1,2,4,8,16]
n = [1024,2048,4096,8192,16384,32768,65536]
import subprocess
subprocess.run("make", shell=True)
print('-'*8)
for i in range(len(num_process)):
    for j in range(len(n)):
        cmd = "./monte_carlo "+str(n[j])+" "+str(num_process[i])
        print('\x1b[32m', cmd, '\x1b[0m')
        subprocess.run(cmd, shell=True)