import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "-p", "--prompt", type=str, help="The prompt for the desired editing", required=True
    )
    parser.add_argument(
        "-i", "--init_image", type=str, help="The path to the source image input", required=True
    )
    parser.add_argument(
        "-i2", "--init_image_2", type=str, help="The path to the source image input", required=True
    )

    parser.add_argument("--mask", type=str, help="The path to the mask to edit with", default=None)

    # Diffusion
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=25,
    )
    parser.add_argument(
        "--local_clip_guided_diffusion",
        help="Indicator for using local CLIP guided diffusion (for baseline comparison)",
        action="store_true",
        dest="local_clip_guided_diffusion",
    )
    parser.add_argument(
        "--ddim",
        help="Indicator for using DDIM instead of DDPM",
        action="store_true",
    )

    # For more details read guided-diffusion/guided_diffusion/respace.py
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="100",
    )
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )

    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)

    # Loss
    parser.add_argument(
        "--clip_guidance_lambda",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=1000,
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=50,
    )
    parser.add_argument(
        "--lpips_sim_lambda",
        type=float,
        help="The LPIPS similarity to the input image",
        default=1000,
    )
    parser.add_argument(
        "--l2_sim_lambda", type=float, help="The L2 similarity to the input image", default=10000,
    )
    parser.add_argument(
        "--background_preservation_loss",
        help="Indicator for using the background preservation loss",
        action="store_true",
    )

    # Mask
    parser.add_argument(
        "--invert_mask",
        help="Indicator for mask inversion",
        action="store_true",
        dest="invert_mask",
    )
    parser.add_argument(
        "--no_enforce_background",
        help="Indicator disabling the last background enforcement",
        action="store_false",
        dest="enforce_background",
    )

    # Misc
    parser.add_argument("--seed", type=int, help="The random seed", default=404)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=8)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=4,
    )
    parser.add_argument(
        "--vid",
        help="Indicator for saving the video of the diffusion process",
        action="store_true",
        dest="save_video",
    )
    parser.add_argument(
        "--export_assets",
        help="Indicator for saving raw assets of the prediction",
        action="store_true",
        dest="export_assets",
    )
    parser.add_argument(
        "--image_guide",
        help="Indicator image or text",
        action="store_true",
        dest="image_guide",
    )
    parser.add_argument(
        "--coarse_to_fine",
        help="Indicator mask from big to small",
        action="store_true",
        dest="coarse_to_fine",
    )
    parser.add_argument(
        "--classifier_scale",
        type=float,
        help="Classifer scale for class guided",
        default=10.,
    )
    parser.add_argument(
        "--y",
        type=int,
        help="Target class for classifier guidence",
        default=0,
    )
    parser.add_argument(
        "--class_cond",
        help="classifer conditioned for diffusion model or not",
        action="store_true",
        dest="class_cond",
    )
    parser.add_argument(
        "--background_complex",
        type=float,
        help="BG complex guidance scale",
        default=0.,
    )
    parser.add_argument(
        "--final_save_root",
        type=str,
        help="Final save root",
        default="validation-generated/generated-with-25-steps-bg/final/",
    )
    parser.add_argument(
        "--hard",
        help="hard or smooth",
        action="store_true",
        dest="hard",
    )
    parser.add_argument(
        "--random_position",
        help="apply random position",
        action="store_true",
        dest="random_position",
    )
    parser.add_argument(
        "--rotate_obj",
        help="apply random rotate to objects",
        action="store_true",
        dest="rotate_obj",
    )
    parser.add_argument(
        "--angle",
        type=int,
        help="angle",
        default=0,
    )
    args = parser.parse_args()
    print(args)
    return args
