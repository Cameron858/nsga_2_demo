import imageio
from pyprojroot import here
from invoke import task


@task
def gif(c, clean=True, fps=24):
    image_dir = here("figures/frames")
    output_dir = here("output")
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / "output.gif"

    # List all PNG files in the directory sorted by their name
    image_files = sorted(
        image_dir.glob("*.png"),
        key=lambda x: int(x.stem.split("_")[-1]),  # Sorting based on generation number
    )

    if not image_files:
        print(f"No PNG files found in {image_dir}")
        return

    # Read images
    images = [imageio.imread(str(image_file)) for image_file in image_files]

    imageio.mimsave(str(gif_path), images, fps=fps)
    print(f"GIF saved to {gif_path} at {fps} fps")

    if clean:
        for f in image_files:
            f.unlink()
        print("Frames deleted.")
