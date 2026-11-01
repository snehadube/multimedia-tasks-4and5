import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def classical_enhance(img):
    """Denoise -> CLAHE (contrast) -> Unsharp mask (sharpen)"""
    den = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.5)
    return cv2.addWeighted(out, 1.5, blur, -0.5, 0)


def ai_upscale(img_bgr, scale=4):
    """EDSR super-resolution (HuggingFace super-image)"""
    import torch
    from super_image import EdsrModel, ImageLoader

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)
    model.eval()
    with torch.no_grad():
        preds = model(ImageLoader.load_image(pil))
    arr = preds.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main(path, max_side, scale):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Image nahi mili: {path}")

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        f = max_side / max(h, w)
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)

    bicubic = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    classical = classical_enhance(img)
    final = ai_upscale(classical, scale)

    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(f"output/{name}_bicubic.png", bicubic)
    cv2.imwrite(f"output/{name}_enhanced.png", final)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, im, t in zip(
        ax,
        [img, bicubic, final],
        [f"Original (sharp={sharpness(img):.0f})",
         f"Bicubic x{scale} (sharp={sharpness(bicubic):.0f})",
         f"Denoise+CLAHE+EDSR x{scale} (sharp={sharpness(final):.0f})"],
    ):
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"output/{name}_comparison.png", dpi=150)
    print("Done. output/ folder check karo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--max_side", type=int, default=400)
    p.add_argument("--scale", type=int, default=4)
    args = p.parse_args()
    main(args.image, args.max_side, args.scale)