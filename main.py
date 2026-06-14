import cv2
import numpy as np
from src.matching import detect_and_match_features


def generate_synthetic_stereo_pair():
    """Generates a synthetic stereo image pair with unique features, noise, and occlusion."""
    base = np.zeros((400, 500), dtype=np.uint8)
    np.random.seed(42)

    # 1. Base features (The "True" environment)
    for _ in range(80):
        x = np.random.randint(0, 500)
        y = np.random.randint(0, 400)
        radius = np.random.randint(2, 15)
        cv2.circle(base, (x, y), radius, 255, -1)

    # Simulate baseline shift
    img_left = base[:, 0:400].copy()
    img_right = base[:, 50:450].copy()

    # 2. Add independent noise to the Left Camera (Distractors)
    for _ in range(20):
        x = np.random.randint(0, 400)
        y = np.random.randint(0, 400)
        cv2.circle(img_left, (x, y), 5, 100, -1)  # Dim grey circles

    # 3. Add independent noise to the Right Camera (Distractors)
    for _ in range(20):
        x = np.random.randint(0, 400)
        y = np.random.randint(0, 400)
        cv2.circle(img_right, (x, y), 5, 100, -1)

    # 4. Simulate Occlusion (Block a portion of the right camera's view)
    cv2.rectangle(img_right, (100, 100), (200, 300), 0, -1)

    return img_left, img_right


def main():
    print("--- Step 3: Executing Feature Matching (Noisy Environment) ---")
    try:
        img_left, img_right = generate_synthetic_stereo_pair()

        kp_left, kp_right, good_matches = detect_and_match_features(img_left, img_right)
        print(f"Found {len(good_matches)} high-quality matches.")

        matched_img = cv2.drawMatches(
            img_left, kp_left,
            img_right, kp_right,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        print("Opening visualization window. Press any key in the window to close it...")
        cv2.imshow("ORB Stereo Matches (With Noise and Occlusion)", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] Failed to execute matching pipeline: {e}")


if __name__ == '__main__':
    main()