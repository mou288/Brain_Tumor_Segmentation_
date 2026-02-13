import os
import cv2
import numpy as np
from skimage.filters import threshold_sauvola


current_dir = os.path.dirname(os.path.abspath(__file__))

base_path = os.path.join(
    current_dir,
    "brain_tumor_dataset",
    "Brain Tumor Segmentation Dataset"
)

image_path = os.path.join(base_path, "image")
mask_path = os.path.join(base_path, "mask")

output_base = os.path.join(current_dir, "results")
os.makedirs(output_base, exist_ok=True)

metrics_file_path = os.path.join(output_base, "metrics.txt")

classes = ["1", "2", "3"]
k_values = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]


def dice_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)


def jaccard_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)


def preprocess(image):
    _, brain_mask = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_mask = brain_mask > 0

    brain_only = np.zeros_like(image)
    brain_only[brain_mask] = image[brain_mask]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  #enhancing the image(increasing contrast)
    brain_enhanced = clahe.apply(brain_only)

    return brain_enhanced, brain_mask


def otsu_segmentation(image, brain_mask):
    brain_pixels = image[brain_mask]
    threshold_value, _ = cv2.threshold(
        brain_pixels.astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    result = np.zeros_like(image, dtype=bool)
    result[brain_mask] = image[brain_mask] > threshold_value
    return result


def sauvola_segmentation(image, brain_mask, k):
    thresh = threshold_sauvola(image, window_size=101, k=k)
    return (image > thresh) & brain_mask


def evaluate(method="otsu", k=None):
    dice_scores = []
    jaccard_scores = []

    for cls in classes:
        img_folder = os.path.join(image_path, cls)
        mask_folder = os.path.join(mask_path, cls)

        for filename in os.listdir(img_folder)[:5]:
            img_file = os.path.join(img_folder, filename)
            name = os.path.splitext(filename)[0]
            mask_file = os.path.join(mask_folder, name + "_m.jpg")

            if not os.path.exists(mask_file):
                continue

            image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                continue

            mask_binary = mask > 128
            brain_enhanced, brain_mask = preprocess(image)

            if method == "otsu":
                pred = otsu_segmentation(brain_enhanced, brain_mask)
            else:
                pred = sauvola_segmentation(brain_enhanced, brain_mask, k)

            dice_scores.append(dice_score(pred, mask_binary))
            jaccard_scores.append(jaccard_score(pred, mask_binary))

    return np.mean(dice_scores), np.mean(jaccard_scores)


mean_otsu_dice, mean_otsu_jaccard = evaluate("otsu")

best_k = None
best_dice = -1

with open(metrics_file_path, "w") as metrics_file:

    metrics_file.write("OTSU:\n")
    metrics_file.write(
        f"Dice: {mean_otsu_dice:.4f}, "
        f"Jaccard: {mean_otsu_jaccard:.4f}\n\n"
    )

    for k_val in k_values:
        mean_dice, mean_jaccard = evaluate("sauvola", k_val)

        metrics_file.write(
            f"Sauvola k={k_val} -> "
            f"Dice: {mean_dice:.4f}, "
            f"Jaccard: {mean_jaccard:.4f}\n"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_k = k_val

    metrics_file.write(
        f"\nBest k: {best_k} "
        f"(Dice = {best_dice:.4f})\n"
    )

print("\nOTSU Dice:", mean_otsu_dice)
print("Best Sauvola k:", best_k, "with Dice:", best_dice)


for cls in classes:

    img_folder = os.path.join(image_path, cls)
    mask_folder = os.path.join(mask_path, cls)

    class_output = os.path.join(output_base, f"class_{cls}")
    os.makedirs(class_output, exist_ok=True)

    count = 0

    for filename in os.listdir(img_folder):

        if count >= 5:
            break

        img_file = os.path.join(img_folder, filename)
        name = os.path.splitext(filename)[0]
        mask_file = os.path.join(mask_folder, name + "_m.jpg")

        if not os.path.exists(mask_file):
            continue

        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        mask_binary = mask > 128
        brain_enhanced, brain_mask = preprocess(image)

        otsu_binary = otsu_segmentation(brain_enhanced, brain_mask)
        sauvola_binary = sauvola_segmentation(brain_enhanced, brain_mask, best_k)

        cv2.imwrite(os.path.join(class_output, f"{name}_original.jpg"), image)
        cv2.imwrite(os.path.join(class_output, f"{name}_mask.jpg"),
                    (mask_binary * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(class_output, f"{name}_otsu.jpg"),
                    (otsu_binary * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(class_output, f"{name}_sauvola_bestk.jpg"),
                    (sauvola_binary * 255).astype(np.uint8))

        count += 1


print("Metrics saved to:", metrics_file_path)
print("Example images saved inside:", output_base)
