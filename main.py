from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.spatial import ConvexHull
import sys

sys.path.append(Path(__file__).parent.as_posix())
from model.create_ChangeDINO import create_model        
from option import Options


def load_image(image, to_tensor, normalize):
    img = image.convert("RGB")
    tensor = normalize(to_tensor(img)).unsqueeze(0)
    return img, tensor


def load_model(dataset_name="S2Looking", gpu_id=0, backbone="mobilenetv2"):
    # Create options programmatically without CLI
    opt = Options.create_from_dict({
        "name": dataset_name,
        "gpu_ids": [gpu_id],
        "backbone": backbone,
    })
    
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
    else:
        raise EnvironmentError("CUDA is not available but gpu_ids were provided.")

    model = create_model(opt)
    model.eval()
    
    return model


def extract_hulls_from_mask(pred_mask):
    """
    Extract convex hulls from a prediction mask.
    
    Args:
        pred_mask: Binary prediction mask (numpy array, H x W)
    
    Returns:
        list: List of convex hulls. Each hull is a list of (x, y) coordinate tuples.
    """
    # Label connected components
    labeled_array, num_features = ndimage.label(pred_mask > 0)
    
    hulls = []
    
    # Extract convex hull for each connected component
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if not component.any():
            continue
        
        # Get coordinates of the component
        y_coords, x_coords = np.where(component)
        
        if len(y_coords) < 3:
            # For small components, just store the points as-is
            if len(y_coords) == 1:
                hull_coords = [(x_coords[0], y_coords[0])]
                hulls.append(hull_coords)
            elif len(y_coords) == 2:
                hull_coords = [(x_coords[0], y_coords[0]), (x_coords[1], y_coords[1])]
                hulls.append(hull_coords)
            continue
        
        # Compute convex hull
        try:
            points = np.column_stack([x_coords, y_coords])
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Convert to list of tuples
            hull_coords = [tuple(pt) for pt in hull_points]
            hulls.append(hull_coords)
            
        except Exception as e:
            # Fallback: if convex hull fails, use bounding box coordinates
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            bbox_coords = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            hulls.append(bbox_coords)
    
    return hulls


def tile_image_change_detection(img1, img2, model):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))

    img_A_pil, img_A = load_image(img1, to_tensor, normalize)
    img_B_pil, img_B = load_image(img2, to_tensor, normalize)

    img_A = img_A.cuda(non_blocking=True)
    img_B = img_B.cuda(non_blocking=True)

    with torch.no_grad():
        pred = model.inference(img_A, img_B)
        pred = torch.argmax(pred, dim=1)
        pred_mask = pred[0].cpu().detach().numpy()
        
        # Extract convex hulls from the mask
        hulls = extract_hulls_from_mask(pred_mask)
        
        return {
            'mask': pred_mask,
            'hulls': hulls,
        }


if __name__ == "__main__":
    model = load_model()
    img_1 = "./input_data/1115_a_old.png"
    img_2 = "./input_data/1115_b_new.png"
    img_1 = Image.open(img_1)
    img_2 = Image.open(img_2)
    result = tile_image_change_detection(img_1, img_2, model)
    print(f"Number of detected changes: {len(result['hulls'])}")
    print(f"Mask shape: {result['mask'].shape}")
