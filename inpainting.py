
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import os

# ===================== YOLO & SD HELPERS =====================

def build_mask_from_yolo(img_bgr, model, target_classes, conf_thres=0.25):
    """
    Use YOLO to detect objects and build a binary mask
    where target_classes are white (255) and rest is black (0).
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    results = model(img_bgr)[0]
    names = model.names

    if results.boxes is None or len(results.boxes) == 0:
        print("No detections from YOLO.")
        return mask

    for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if score < conf_thres:
            continue

        class_name = names[int(cls_id)]
        if class_name not in target_classes:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def prepare_mask(raw_mask, dilate_iter=6):
    """
    Clean and slightly expand mask for better inpainting.
    Also good for catching shadows around balls.
    """
    _, m = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        m = cv2.dilate(m, kernel, iterations=dilate_iter)
    return m


def load_sd_inpaint_pipeline(device: str):
    """
    Load Stable Diffusion inpainting pipeline.
    Model: runwayml/stable-diffusion-inpainting
    """
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
       type=dtype,
    )
    pipe = pipe.to(device)
    return pipe


def run_sd_inpaint(base_img_bgr, mask, prompt, negative_prompt,
                   guidance_scale=7.5, num_inference_steps=45):
    """
    Run Stable Diffusion inpainting once with given mask.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pipe = load_sd_inpaint_pipeline(device)

    img_rgb = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)
    mask_pil = Image.fromarray(mask)

    print("Running Stable Diffusion inpainting...")
    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    result_np = np.array(result)  # RGB
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    return result_bgr

# ===================== INTERACTIVE MASK EDITOR =====================

# Global state for editor
img = None          # original image (BGR)
mask = None         # binary mask (uint8, 0/255)
work_img = None     # image we display (usually original)

WIN_W = WIN_H = 0

current_mode = "brush"   # 'brush', 'erase', 'hand'
brush_size = 20
drawing = False
start_point = None        # for brush lines (image coords)
current_point = None

undo_stack = []
redo_stack = []
MAX_HISTORY = 50

zoom = 1.0
MIN_ZOOM = 1.0
MAX_ZOOM = 8.0
view_cx = 0.0
view_cy = 0.0
hand_last_mouse = None


def update_window_title():
    title = (
        f"Mode: {current_mode.upper()} | "
        "[B]rush [E]rase [H]and | [Z/X] Zoom In/Out | "
        "[U]ndo [Y]redo | +/- Brush size | Enter=Inpaint, Q=Quit"
    )
    try:
        cv2.setWindowTitle("YOLO Mask Editor", title)
    except Exception:
        pass


def push_undo(prev_mask):
    global undo_stack, redo_stack
    if prev_mask is None:
        return
    if len(undo_stack) >= MAX_HISTORY:
        undo_stack.pop(0)
    undo_stack.append(prev_mask.copy())
    redo_stack.clear()


def clamp_view_center():
    global view_cx, view_cy
    view_w = int(round(WIN_W / zoom))
    view_h = int(round(WIN_H / zoom))
    view_w = min(view_w, img.shape[1])
    view_h = min(view_h, img.shape[0])

    half_w = view_w / 2.0
    half_h = view_h / 2.0

    view_cx = max(half_w, min(img.shape[1] - half_w, view_cx))
    view_cy = max(half_h, min(img.shape[0] - half_h, view_cy))


def get_view_params():
    clamp_view_center()
    view_w = int(round(WIN_W / zoom))
    view_h = int(round(WIN_H / zoom))
    view_w = min(view_w, img.shape[1])
    view_h = min(view_h, img.shape[0])

    vx = int(round(view_cx - view_w / 2.0))
    vy = int(round(view_cy - view_h / 2.0))

    vx = max(0, min(img.shape[1] - view_w, vx))
    vy = max(0, min(img.shape[0] - view_h, vy))

    return vx, vy, view_w, view_h


def window_to_image(x_win, y_win):
    vx, vy, vw, vh = get_view_params()

    x_win = max(0, min(WIN_W - 1, x_win))
    y_win = max(0, min(WIN_H - 1, y_win))

    img_x = vx + (x_win / float(WIN_W)) * vw
    img_y = vy + (y_win / float(WIN_H)) * vh

    img_x = int(round(img_x))
    img_y = int(round(img_y))

    img_x = max(0, min(img.shape[1] - 1, img_x))
    img_y = max(0, min(img.shape[0] - 1, img_y))

    return img_x, img_y


def build_display():
    """
    Overlay mask in red over work_img, then crop+zoom.
    """
    base = work_img.copy()
    red_overlay = work_img.copy()
    red_overlay[mask == 255] = (0, 0, 255)
    display_full = cv2.addWeighted(base, 0.7, red_overlay, 0.3, 0)

    vx, vy, vw, vh = get_view_params()
    crop = display_full[vy:vy + vh, vx:vx + vw]
    display = cv2.resize(crop, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
    return display


def mouse_callback(event, x_win, y_win, flags, param):
    global drawing, start_point, current_point
    global hand_last_mouse, view_cx, view_cy, mask

    # Hand tool
    if current_mode == "hand":
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            hand_last_mouse = (x_win, y_win)
        elif event == cv2.EVENT_MOUSEMOVE and drawing and hand_last_mouse is not None:
            dx_win = x_win - hand_last_mouse[0]
            dy_win = y_win - hand_last_mouse[1]
            vx, vy, vw, vh = get_view_params()
            dx_img = (dx_win / float(WIN_W)) * vw
            dy_img = (dy_win / float(WIN_H)) * vh
            view_cx -= dx_img
            view_cy -= dy_img
            clamp_view_center()
            hand_last_mouse = (x_win, y_win)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            hand_last_mouse = None
        return

    # Map window → image coords
    img_x, img_y = window_to_image(x_win, y_win)
    img_pt = (img_x, img_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = img_pt
        current_point = img_pt
        # store undo snapshot
        push_undo(mask.copy())

        if current_mode == "brush":
            cv2.circle(mask, img_pt, brush_size // 2, 255, -1)
        elif current_mode == "erase":
            cv2.circle(mask, img_pt, brush_size // 2, 0, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        current_point = img_pt
        if drawing:
            if current_mode == "brush":
                cv2.line(mask, start_point, img_pt, 255, brush_size)
                start_point = img_pt
            elif current_mode == "erase":
                cv2.line(mask, start_point, img_pt, 0, brush_size)
                start_point = img_pt

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = img_pt
        if current_mode == "brush":
            cv2.line(mask, start_point, img_pt, 255, brush_size)
        elif current_mode == "erase":
            cv2.line(mask, start_point, img_pt, 0, brush_size)


# ===================== ARGPARSE & MAIN =====================

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + SD Inpainting with interactive mask editor")
    parser.add_argument("--image", "-i", type=str, default="image.jpeg", help="Input image path")
    parser.add_argument("--output", "-o", type=str, default="output_inpainted.png", help="Output image path")
    parser.add_argument(
        "--classes", "-c", type=str, nargs="+",
        default=["sports ball"],
        help="YOLO class names to remove (e.g. 'sports ball' 'person')"
    )
    parser.add_argument(
        "--prompt", "-p", type=str,
        default="empty cricket pitch, no balls, no equipment, no shadows, only natural pitch and grass, highly detailed, realistic",
        help="Positive prompt for Stable Diffusion"
    )
    parser.add_argument(
        "--negative", "-n", type=str,
        default="balls, cricket ball, sports ball, red objects, round objects, circles, spheres, shadows, blur, artifacts, distortions",
        help="Negative prompt for Stable Diffusion"
    )
    parser.add_argument(
        "--steps", type=int, default=45,
        help="Number of diffusion steps (higher = better but slower)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--dilate", type=int, default=6,
        help="Dilation iterations for initial YOLO mask"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- Load image ---
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    img = img_bgr
    work_img = img_bgr.copy()
    h, w = img.shape[:2]
    WIN_W, WIN_H = w, h
    view_cx, view_cy = w / 2.0, h / 2.0

    # --- YOLO initial mask ---
    print("Loading YOLO model...")
    yolo_model = YOLO("yolov8n.pt")
    print(f"Running YOLO for initial mask on classes: {args.classes}")
    raw_mask = build_mask_from_yolo(img, yolo_model, args.classes, conf_thres=args.conf)
    mask = prepare_mask(raw_mask, dilate_iter=args.dilate)

    # --- Interactive editor ---
    cv2.namedWindow("YOLO Mask Editor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Mask Editor", WIN_W, WIN_H)
    cv2.setMouseCallback("YOLO Mask Editor", mouse_callback)
    update_window_title()

    print("""
Interactive YOLO Mask Editor

You are now seeing the YOLO-selected areas in RED.
Refine the mask, then press Enter to run high-quality AI inpainting.

Controls:
  B - Brush (add to mask)
  E - Erase (remove from mask)
  H - Hand (hold LMB and drag to pan when zoomed)
  Z - Zoom in
  X - Zoom out
  U - Undo (mask)
  Y - Redo (mask)
  + / - - Increase / Decrease brush size
  Enter - Run Stable Diffusion inpainting with current mask
  Q - Quit without inpainting
""")

    while True:
        disp = build_display()
        cv2.imshow("YOLO Mask Editor", disp)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # Enter: run SD inpainting
            clean_mask = prepare_mask(mask, dilate_iter=0)  # already edited
            result = run_sd_inpaint(
                base_img_bgr=img,
                mask=clean_mask,
                prompt=args.prompt,
                negative_prompt=args.negative,
                num_inference_steps=args.steps,
            )
            cv2.imwrite(args.output, result)
            print(f"Saved inpainted image to: {args.output}")
            cv2.imshow("Inpaint Result", result)
            cv2.waitKey(0)
            break

        elif key in (ord('q'), ord('Q')):
            break

        # Tool switching
        elif key in (ord('b'), ord('B')):
            current_mode = "brush"
            update_window_title()
        elif key in (ord('e'), ord('E')):
            current_mode = "erase"
            update_window_title()
        elif key in (ord('h'), ord('H')):
            current_mode = "hand"
            update_window_title()

        # Undo / Redo
        elif key in (ord('u'), ord('U')):
            if undo_stack:
                redo_stack.append(mask.copy())
                mask[:] = undo_stack.pop()
                print("Undo")
        elif key in (ord('y'), ord('Y')):
            if redo_stack:
                undo_stack.append(mask.copy())
                mask[:] = redo_stack.pop()
                print("Redo")

        # Brush size
        elif key in (ord('+'), ord('=')):
            brush_size = min(brush_size + 2, 200)
            print("Brush size:", brush_size)
        elif key in (ord('-'), ord('_')):
            brush_size = max(3, brush_size - 2)
            print("Brush size:", brush_size)

        # Zoom
        elif key in (ord('z'), ord('Z')):
            zoom = min(MAX_ZOOM, zoom * 1.25)
            clamp_view_center()
            print("Zoom:", round(zoom, 2))
        elif key in (ord('x'), ord('X')):
            zoom = max(MIN_ZOOM, zoom / 1.25)
            clamp_view_center()
            print("Zoom:", round(zoom, 2))

    cv2.destroyAllWindows()
