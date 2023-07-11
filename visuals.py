import os
import sys
import subprocess
import imageio.v2 as imageio

def count_frames(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_frames = 0
    for line in lines:
        line = line.strip()
        if line.isdigit():
            num_atoms = int(line)
            num_frames += 1
        elif num_atoms and len(line.split()) == 4:
            num_atoms -= 1
    return num_frames


def create_movie(input_file, root):
    output_file = input_file.rsplit('.', 1)[0]
    image_dir = f'{output_file}_images'
    os.makedirs(image_dir, exist_ok=True)

    vmd_script = f"""
    mol new {input_file} type xyz
    mol rename top trajectory
    display projection Orthographic
    display depthcue off
    material glossy

    # Set CPK representation for atoms
    mol representation CPK 0.8 0.1 50 50
    mol material Glossy

    mol addrep top
    mol drawframes top 0 {count_frames(input_file)}
    mol modcolor 0 top Element

    # Set the background color to white
    color Display Background white

    # Remove the axis
    axes location Off

    # Loop over frames and render each frame
    set num_frames {count_frames(input_file)}
    for {{set i 0}} {{$i < $num_frames}} {{incr i}} {{
        animate goto $i
        render TachyonInternal {image_dir}/frame_$i.tga
    }}

    quit
    """

    vmd_command = ['vmd', '-dispdev', 'text', '-e', f'{root}/vmd_script.tcl']
    with open(f'{root}/vmd_script.tcl', 'w') as f:
        f.write(vmd_script)

    subprocess.run(vmd_command)
    os.remove(f'{root}/vmd_script.tcl')

    # Convert the rendered images to PNG using imageio
    images = []
    num_frames = len(os.listdir(image_dir))

    for i in range(num_frames):
        image_path = os.path.join(image_dir, f'frame_{i}.tga')
        image = imageio.imread(image_path)
        images.append(image)

    movie_file = f'{output_file}.mp4'
    imageio.mimsave(movie_file, images, fps=5)

    # Remove the image directory
    for file in os.listdir(image_dir):
        os.remove(os.path.join(image_dir, file))
    os.rmdir(image_dir)


def process_trajectories(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xyz'):
                input_file = os.path.join(root, file)
                create_movie(input_file, root)


if __name__ == "__main__":
    # Check if the directory argument is provided
    if len(sys.argv) < 2:
        print("Please provide the directory path as an argument.")
        sys.exit(1)
    
    # Get the directory path from command line argument
    directory = sys.argv[1]
    
    # Process the trajectories
    process_trajectories(directory)

