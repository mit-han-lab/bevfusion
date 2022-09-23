import os 

path_to_bevfusion = '/ABSOLUTE/PATH/TO/BEVFUSION' #path outside of docker 
docker_path_bevfusion = path_to_bevfusion         #path in docker
path_to_nuscenes = '/ABSOLUTE/PATH/TO/NUSCENES'   #to make symbolic link to data work


memory = '724g'
shm_size = '362g'
cpus = 72
gpus = 'all'
port = 4405
image = 'benjamintherien/dev:cugl11.1.1-py3.8-minimal-torch1.9.0-mmlab-bevfusion'
name = 'bevfusion'

command = "docker run -v {}:{} -v {}:{} --memory {} --cpus={} --gpus {} --shm-size {} -p {}:{} --name {} --rm -it {}".format(
    path_to_bevfusion,docker_path_bevfusion,path_to_nuscenes,path_to_nuscenes,memory,cpus,gpus,shm_size,port,port,name,image
)

print("################################################")
print('[run_docker.py executing] ',command)
print("################################################")

os.system(command)