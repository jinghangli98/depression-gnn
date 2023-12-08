import cv2
import os
from natsort import natsorted
def generate_movie(images_folder, output_movie_path, fps=5):
    """
    Generate a movie from a sequence of PNG images.

    Parameters:
    - images_folder (str): Path to the folder containing PNG images.
    - output_movie_path (str): Path to the output movie file (e.g., 'output.mp4').
    - fps (int): Frames per second for the output video (default is 30).
    """
    # Get the list of PNG images in the specified folder
    image_files = natsorted([f for f in os.listdir(images_folder) if f.endswith('.png')])

    if not image_files:
        print("No PNG images found in the specified folder.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(images_folder, image_files[0]))
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format
    video_writer = cv2.VideoWriter(output_movie_path, fourcc, fps, (width, height))

    # Write each image to the video file
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video generated successfully at {output_movie_path}")

# Example usage:
images_folder_path = './figure/train/'
output_movie_path = 'train.mp4'
generate_movie(images_folder_path, output_movie_path)
