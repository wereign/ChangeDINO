import os
from argparse import ArgumentParser

import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from scipy import ndimage
from scipy.spatial import ConvexHull

from model.create_ChangeDINO import create_model
from option import Options


def build_parser() -> ArgumentParser:
    """
    Reuse the training options so checkpoints/backbone configs stay in sync,
    and extend with paths for single-pair inference.
    """
    opt_builder = Options()
    opt_builder.init()
    parser = opt_builder.parser
    # parser.set_defaults(name="SYSU-CD")
    parser.set_defaults(name="S2Looking")
    # parser.set_defaults(name="LEVIR-CD")
    parser.add_argument("--img_A", required=True, help="Path to time-A image.")
    parser.add_argument("--img_B", required=True, help="Path to time-B image.")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/run_pred.png",
        help="Where to save the binary prediction mask (0/255).",
    )
    return parser


def parse_and_prepare() -> object:
    parser = build_parser()
    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(",")
    opt.gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            opt.gpu_ids.append(gid)
    if not opt.gpu_ids:
        raise ValueError("gpu_ids must include at least one GPU id (e.g., 0).")

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
    else:
        raise EnvironmentError("CUDA is not available but gpu_ids were provided.")

    opt.phase = "test"
    opt.load_pretrain = True
    opt.batch_size = 1
    opt.num_workers = 0

    print("------------ Options -------------")
    for k, v in sorted(vars(opt).items()):
        print(f"{k}: {v}")
    print("-------------- End ----------------")

    return opt


def load_image(path, to_tensor, normalize):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist.")
    img = Image.open(path).convert("RGB")
    tensor = normalize(to_tensor(img)).unsqueeze(0)
    return img, tensor


def extract_and_draw_bboxes(pred_mask, img_B_pil, output_path, bbox_color="red", bbox_width=3):
    """
    Extract convex hulls from prediction mask and draw them on img_B.
    
    Args:
        pred_mask: Binary prediction mask (numpy array, H x W)
        img_B_pil: PIL Image of the second image
        output_path: Path to save the output image with convex hulls
        bbox_color: Color of convex hulls (default: "red")
        bbox_width: Width of hull lines (default: 3)
    """
    # Label connected components
    labeled_array, num_features = ndimage.label(pred_mask > 0)
    
    # Create a copy of img_B to draw on
    img_with_hull = img_B_pil.copy()
    draw = ImageDraw.Draw(img_with_hull)
    
    hulls = []
    
    # Extract convex hull for each connected component
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if not component.any():
            continue
        
        # Get coordinates of the component
        y_coords, x_coords = np.where(component)
        
        if len(y_coords) < 3:
            # Need at least 3 points for a convex hull, otherwise draw a point or line
            if len(y_coords) == 1:
                x, y = x_coords[0], y_coords[0]
                draw.point((x, y), fill=bbox_color)
            elif len(y_coords) == 2:
                points = [(x_coords[0], y_coords[0]), (x_coords[1], y_coords[1])]
                draw.line(points, fill=bbox_color, width=bbox_width)
            continue
        
        # Compute convex hull
        try:
            points = np.column_stack([x_coords, y_coords])
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Convert to list of tuples for PIL
            hull_coords = [tuple(pt) for pt in hull_points]
            hulls.append(hull_coords)
            
            # Draw polygon on image (close the hull by adding the first point at the end)
            hull_coords_closed = hull_coords + [hull_coords[0]]
            draw.polygon(hull_coords, outline=bbox_color, width=bbox_width)
            
        except Exception as e:
            # Fallback: if convex hull fails, draw axis-aligned bbox
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            if x_min < x_max and y_min < y_max:
                draw.rectangle((x_min, y_min, x_max, y_max), outline=bbox_color, width=bbox_width)
    
    # Save the image with hulls
    img_with_hull.save(output_path)
    print(f"Saved image with {len(hulls)} convex hulls to {output_path}")
    
    return hulls


def main():
    opt = parse_and_prepare()

    # Create output directory organized by dataset name
    dataset_output_dir = os.path.join("./outputs", opt.name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Generate unique output filename based on input images
    img_a_basename = os.path.splitext(os.path.basename(opt.img_A))[0]
    img_b_basename = os.path.splitext(os.path.basename(opt.img_B))[0]
    output_basename = f"{img_a_basename}_vs_{img_b_basename}"
    
    pred_output = os.path.join(dataset_output_dir, f"{output_basename}_mask.png")
    bbox_output_a = os.path.join(dataset_output_dir, f"{output_basename}_old_hull.png")
    bbox_output_b = os.path.join(dataset_output_dir, f"{output_basename}_new_hull.png")
    
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))

    img_A_pil, img_A = load_image(opt.img_A, to_tensor, normalize)
    img_B_pil, img_B = load_image(opt.img_B, to_tensor, normalize)

    img_A = img_A.cuda(non_blocking=True)
    img_B = img_B.cuda(non_blocking=True)

    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        pred = model.inference(img_A, img_B)
        pred = torch.argmax(pred, dim=1)
        pred_mask = pred[0].cpu().detach().numpy()
        
        # Save binary prediction mask
        pred_img = Image.fromarray((pred_mask * 255).astype("uint8"))
        pred_img.save(pred_output)
        
        print(f"Saved prediction mask to {pred_output}")
        
        # Extract and draw convex hulls on both old and new images
        extract_and_draw_bboxes(pred_mask, img_A_pil, bbox_output_a)
        extract_and_draw_bboxes(pred_mask, img_B_pil, bbox_output_b)


if __name__ == "__main__":
    main()
