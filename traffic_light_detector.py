"""
Traffic Light Detection System
ITS 301 - Artificial Intelligence
Assignment 2 - Programming Implementation

This program detects traffic light colours (Red, Yellow, Green) from images
using OpenCV colour-based detection with HSV thresholding, and determines
the appropriate driving action.
"""

import cv2
import numpy as np
import os
import sys


# ──────────────────────────────────────────────
#  COLOUR THRESHOLDS (HSV colour space)
# ──────────────────────────────────────────────

# Red wraps around the HSV hue circle, so two ranges are needed
RED_LOWER_1  = np.array([0,   120, 70])
RED_UPPER_1  = np.array([10,  255, 255])
RED_LOWER_2  = np.array([160, 120, 70])
RED_UPPER_2  = np.array([180, 255, 255])

YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

GREEN_LOWER  = np.array([36, 80,  50])
GREEN_UPPER  = np.array([90, 255, 255])

# Minimum pixel area to count as a detected colour region
MIN_AREA = 200


# ──────────────────────────────────────────────
#  CORE DETECTION FUNCTIONS
# ──────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk. Raises FileNotFoundError if path is invalid.

    Args:
        image_path: Path to the image file.

    Returns:
        BGR image as a NumPy array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to HSV colour space for colour-based detection.
    Applies a slight Gaussian blur to reduce noise before conversion.

    Args:
        image: BGR image array.

    Returns:
        HSV image array.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv


def count_colour_pixels(hsv_image: np.ndarray, lower: np.ndarray,
                        upper: np.ndarray,
                        lower2: np.ndarray = None,
                        upper2: np.ndarray = None) -> int:
    """
    Count the number of pixels within a given HSV colour range.
    Supports a second range for colours like red that wrap the hue axis.

    Args:
        hsv_image:  HSV image array.
        lower:      Lower bound of the primary HSV range.
        upper:      Upper bound of the primary HSV range.
        lower2:     (Optional) Lower bound of secondary HSV range.
        upper2:     (Optional) Upper bound of secondary HSV range.

    Returns:
        Total pixel count matching the colour range(s).
    """
    mask = cv2.inRange(hsv_image, lower, upper)

    if lower2 is not None and upper2 is not None:
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2)

    return int(np.sum(mask > 0))


def detect_traffic_light_colour(image: np.ndarray) -> tuple[str, int]:
    """
    Detect the dominant traffic light colour in an image.

    The function analyses the upper third of the image (where traffic lights
    typically appear) and returns the colour with the highest pixel count,
    provided it exceeds MIN_AREA.

    Args:
        image: BGR image array (full frame).

    Returns:
        Tuple of (colour_label, pixel_count).
        colour_label is one of: "RED", "YELLOW", "GREEN", "UNKNOWN".
    """
    # Focus on the upper portion of the image
    height = image.shape[0]
    region_of_interest = image[:int(height * 0.75), :]

    hsv = preprocess_image(region_of_interest)

    red_pixels    = count_colour_pixels(hsv, RED_LOWER_1,  RED_UPPER_1,
                                             RED_LOWER_2,  RED_UPPER_2)
    yellow_pixels = count_colour_pixels(hsv, YELLOW_LOWER, YELLOW_UPPER)
    green_pixels  = count_colour_pixels(hsv, GREEN_LOWER,  GREEN_UPPER)

    scores = {
        "RED":    red_pixels,
        "YELLOW": yellow_pixels,
        "GREEN":  green_pixels,
    }

    dominant_colour = max(scores, key=scores.get)
    dominant_count  = scores[dominant_colour]

    if dominant_count < MIN_AREA:
        return "UNKNOWN", dominant_count

    return dominant_colour, dominant_count


def determine_action(colour: str) -> str:
    """
    Map a detected traffic light colour to the appropriate driving action.

    Args:
        colour: One of "RED", "YELLOW", "GREEN", "UNKNOWN".

    Returns:
        A string describing the driving action.
    """
    action_map = {
        "RED":     "STOP",
        "YELLOW":  "SLOW DOWN",
        "GREEN":   "GO",
        "UNKNOWN": "PROCEED WITH CAUTION",
    }
    return action_map.get(colour, "PROCEED WITH CAUTION")


# ──────────────────────────────────────────────
#  VISUALISATION HELPER
# ──────────────────────────────────────────────

def annotate_image(image: np.ndarray, colour: str, action: str,
                   pixel_count: int) -> np.ndarray:
    """
    Draw a colour-coded overlay and label onto the image.

    Args:
        image:       Original BGR image.
        colour:      Detected colour label.
        action:      Driving action string.
        pixel_count: Number of matching pixels detected.

    Returns:
        Annotated BGR image.
    """
    colour_bgr_map = {
        "RED":     (0,   0,   220),
        "YELLOW":  (0,   220, 220),
        "GREEN":   (0,   200, 50),
        "UNKNOWN": (150, 150, 150),
    }
    bgr = colour_bgr_map.get(colour, (150, 150, 150))

    annotated = image.copy()

    # Semi-transparent banner at bottom
    overlay = annotated.copy()
    h, w = annotated.shape[:2]
    cv2.rectangle(overlay, (0, h - 70), (w, h), bgr, -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # Text labels
    cv2.putText(annotated, f"Colour: {colour}",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Action: {action}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Pixels: {pixel_count}",
                (w - 160, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return annotated


# ──────────────────────────────────────────────
#  SYNTHETIC TEST IMAGE GENERATOR
# ──────────────────────────────────────────────

def create_synthetic_traffic_light(colour: str, filename: str) -> str:
    """
    Generate a synthetic traffic light image for testing purposes.
    Creates a realistic-looking traffic light housing with the correct
    light illuminated.

    Args:
        colour:   One of "RED", "YELLOW", "GREEN".
        filename: Output file path (PNG).

    Returns:
        The filename path.
    """
    img = np.zeros((300, 150, 3), dtype=np.uint8)

    # Dark background
    img[:] = (30, 30, 30)

    # Traffic light housing (dark grey rectangle)
    cv2.rectangle(img, (30, 20), (120, 280), (60, 60, 60), -1)
    cv2.rectangle(img, (30, 20), (120, 280), (90, 90, 90),  2)

    # Define circle positions for R / Y / G
    light_positions = {
        "RED":    [(75, 70), (75, 150), (75, 230)],   # top lit
        "YELLOW": [(75, 70), (75, 150), (75, 230)],   # middle lit
        "GREEN":  [(75, 70), (75, 150), (75, 230)],   # bottom lit
    }
    active_index = {"RED": 0, "YELLOW": 1, "GREEN": 2}

    # Dim colour for inactive lights
    dim_colours = {
        "RED":    (0,  0,  60),
        "YELLOW": (0, 60,  60),
        "GREEN":  (0, 60,   0),
    }
    # Bright colour for active light
    active_colours = {
        "RED":    (0,  0,  255),
        "YELLOW": (0, 220, 220),
        "GREEN":  (0, 220,  30),
    }

    positions = light_positions.get(colour, light_positions["RED"])
    active_i  = active_index.get(colour, 0)

    for light_colour, pos in zip(["RED", "YELLOW", "GREEN"], positions):
        is_active = (light_colour == colour)
        fill_colour = active_colours[light_colour] if is_active \
                      else dim_colours[light_colour]
        cv2.circle(img, pos, 28, fill_colour, -1)

        # Add a bright centre spot to the active light
        if is_active:
            cv2.circle(img, pos, 14, tuple(min(c + 80, 255) for c in fill_colour), -1)

    cv2.imwrite(filename, img)
    return filename


# ──────────────────────────────────────────────
#  BATCH PROCESSING
# ──────────────────────────────────────────────

def process_image(image_path: str, save_output: bool = True) -> dict:
    """
    Full pipeline: load → detect → annotate → (optionally save) one image.

    Args:
        image_path:   Path to the input image.
        save_output:  Whether to save the annotated image.

    Returns:
        Dictionary with keys: path, colour, action, pixel_count, output_path.
    """
    try:
        image = load_image(image_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [ERROR] {exc}")
        return {"path": image_path, "colour": "ERROR", "action": "N/A",
                "pixel_count": 0, "output_path": None}

    colour, pixel_count = detect_traffic_light_colour(image)
    action = determine_action(colour)

    output_path = None
    if save_output:
        annotated = annotate_image(image, colour, action, pixel_count)
        base, ext  = os.path.splitext(image_path)
        output_path = f"{base}_result{ext}"
        cv2.imwrite(output_path, annotated)

    return {
        "path":        image_path,
        "colour":      colour,
        "action":      action,
        "pixel_count": pixel_count,
        "output_path": output_path,
    }


def run_test_suite(test_dir: str = "test_images") -> None:
    """
    Generate 5 synthetic test images (RED ×2, YELLOW ×1, GREEN ×2),
    run detection on each, and print a results summary.

    Args:
        test_dir: Directory to store generated test images.
    """
    os.makedirs(test_dir, exist_ok=True)

    # Create 5 synthetic test images
    test_cases = [
        ("RED",    "test_01_red.png"),
        ("GREEN",  "test_02_green.png"),
        ("YELLOW", "test_03_yellow.png"),
        ("RED",    "test_04_red_2.png"),
        ("GREEN",  "test_05_green_2.png"),
    ]

    print("\n" + "=" * 55)
    print("  TRAFFIC LIGHT DETECTION SYSTEM — TEST SUITE")
    print("=" * 55)

    correct = 0
    total   = len(test_cases)

    for expected_colour, fname in test_cases:
        img_path = os.path.join(test_dir, fname)

        # Generate synthetic image
        create_synthetic_traffic_light(expected_colour, img_path)

        # Run detection
        result = process_image(img_path, save_output=True)

        detected = result["colour"]
        action   = result["action"]
        pixels   = result["pixel_count"]
        status   = "✓ PASS" if detected == expected_colour else "✗ FAIL"

        if detected == expected_colour:
            correct += 1

        print(f"\n  Image   : {fname}")
        print(f"  Expected: {expected_colour}")
        print(f"  Detected: {detected}  (pixels: {pixels})")
        print(f"  Action  : {action}")
        print(f"  Status  : {status}")
        if result["output_path"]:
            print(f"  Saved   : {result['output_path']}")

    accuracy = (correct / total) * 100
    print("\n" + "-" * 55)
    print(f"  RESULTS: {correct}/{total} correct  |  Accuracy: {accuracy:.1f}%")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process a specific image passed as a command-line argument
        path = sys.argv[1]
        print(f"\nProcessing: {path}")
        result = process_image(path)
        print(f"  Colour  : {result['colour']}")
        print(f"  Action  : {result['action']}")
        print(f"  Pixels  : {result['pixel_count']}")
    else:
        # Run the built-in test suite with synthetic images
        run_test_suite()
