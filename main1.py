import cv2
import numpy as np

def segment_image_kmeans_side_by_side(image_path, K_values=[2, 3, 4, 5, 6, 7]):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 400))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    results = [img]

    for K in K_values:
        _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(img_rgb.shape)
        segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        labeled = segmented_bgr.copy()
        cv2.putText(labeled, f"K = {K}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        results.append(labeled)

    combined = cv2.hconcat(results)

    cv2.imshow("K-means segmentatsiya (yonma-yon)", combined)

    cv2.imwrite("segment_yonmayon.jpg", combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

segment_image_kmeans_side_by_side("img_2.png", K_values=[2, 3, 4, 5, 6, 7])