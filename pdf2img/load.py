import cv2
import numpy as np

img = cv2.imread("images/page_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

denoised = cv2.fastNlMeansDenoising(gray, h=30)

thresh = cv2.adaptiveThreshold(
    denoised, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

(h, w) = thresh.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
deskew = cv2.warpAffine(thresh, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
processed = cv2.morphologyEx(deskew, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("processed.png", processed)