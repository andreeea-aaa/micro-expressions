import os
import subprocess

REAL_DIR = "/Users/andreeabrad/Downloads/Extractions_DFD/DFD_original_sequences"
FAKE_DIR = "/Users/andreeabrad/Downloads/Extractions_DFD/DFD_manipulated_sequences"
OUTPUT_DIR = "/Users/andreeabrad/Downloads/Extractions_DFD/DFD_processed"
CONTAINER_NAME = "vigilant_visvesvaraya"
CONTAINER_WORKDIR = "/home/openface-build/build/bin"

video_names = {
    "real": [x for x in os.listdir(REAL_DIR) if x.endswith('.mp4')],
    "fake": [x for x in os.listdir(FAKE_DIR) if x.endswith('.mp4')]
}

total_videos = len(video_names["fake"])
counter = 0

def copy_folder_to_container(local_path, container_path):
    cmd = ["docker", "cp", local_path, f"{CONTAINER_NAME}:{container_path}"]
    subprocess.run(cmd, check=True)


def docker_exec(commands):
    cmd = ["docker", "exec", CONTAINER_NAME, "/bin/bash", "-c", commands]
    subprocess.run(cmd, check=True)


def process_videos(type):
    global counter
    all_videos = video_names[type]

    container_target = f"{CONTAINER_WORKDIR}/{type}"

    for video_name in all_videos:
        counter += 1
        print(f"Processing {video_name} ({counter}/{total_videos})")

        commands = f"cd {CONTAINER_WORKDIR} && ./FeatureExtraction -f {container_target}/{video_name}"
        docker_exec(commands)



    csv_files = [video_path.replace('.mp4', '.csv') for video_path in all_videos]

    for csv_file in csv_files:
        docker_cp_cmd = [
            "docker", "cp",
            f"{CONTAINER_NAME}:{CONTAINER_WORKDIR}/processed/{csv_file}",
            f"{OUTPUT_DIR}/{type}/{csv_file}"
        ]
        subprocess.run(docker_cp_cmd, check=True)

copy_folder_to_container(REAL_DIR, CONTAINER_WORKDIR + "/real")
copy_folder_to_container(FAKE_DIR, CONTAINER_WORKDIR + "/fake")


process_videos("real")
process_videos("fake")