# clear; close all; clc;
import os
import glob
import time
import cv2
# import numpy as np
# import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # work from the script's folder

# %% Settings
ZOOM_PCT = 70        # 1.7 zoom 
saveDir = 'captures'

def zoom(img):
    h, w = img.shape[:2]
    s = 1 + ZOOM_PCT / 100
    ch, cw = round(h / s), round(w / s)
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC)

# %% Menu
print('1. Acquire from Webcam')
print('2. Read from File')
choice = input('Choose option:  ').strip()

if choice == '1':
    # %% Acquire from webcam, SPACE = take picture, q = quit
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit('Could not open camera')
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # fixed focus
    while True:
        # asked ai to help me create settings for capture from webcam
        ok, f = cap.read()
        if not ok:
            raise SystemExit('Camera gave no frame')
        f = zoom(f)
        cv2.putText(f, 'SPACE = take picture   q = quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Preview', f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            for _ in range(10):
                ok, f = cap.read()
            f = zoom(f)
            break
        if key == ord('q'):
            raise SystemExit('No picture taken')
    cap.release()
    cv2.destroyAllWindows()

    os.makedirs(saveDir, exist_ok=True)
    cv2.imwrite(time.strftime(saveDir + '/card_%Y%m%d_%H%M%S.png'), f)

elif choice == '2':
    # %% Read from file
    names = [n for e in ('png', 'jpg', 'jpeg', 'tif', 'tiff')
             for n in glob.glob(saveDir + '/*.' + e) + glob.glob('*.' + e)]
    if not names:
        raise SystemExit('No image found')
    for k, n in enumerate(names, 1):
        print('%d: %s' % (k, n))
    f = cv2.imread(names[int(input('Choose image from above:  ')) - 1])

else:
    raise SystemExit('Invalid option')

# # %% Even lighting
# low-pass the image to estimate the lighting, then subtract it (same as P3)
# g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
# F = np.fft.fftshift(np.fft.fft2(g))

# Frequency coordinates (cycles/image) and distance from the center
# M, N = F.shape
# u, v = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
# U = u - (N // 2 + 1)
# V = v - (M // 2 + 1)
# D = np.sqrt(U**2 + V**2)

# Butterworth low-pass
# n = 2    # order
# D0 = 10  # cutoff
# HL = 1 / (1 + (D / D0)**(2 * n))
# light = np.real(np.fft.ifft2(np.fft.ifftshift(F * HL)))   # estimate of the lighting
# h3 = g - light

# # %% Output (comment this whole block out for the final project)
# cx = U[0, :]
# cy = V[:, 0]

# Filter
# plt.figure()
# plt.imshow(HL, extent=[cx[0], cx[-1], cy[-1], cy[0]], cmap='gray', vmin=0, vmax=1)
# plt.colorbar()
# plt.xlim(-50, 50); plt.ylim(50, -50)
# plt.xlabel('Horizontal frequency (cycles/image)')
# plt.ylabel('Vertical frequency (cycles/image)')
# plt.title('Filter Used to Estimate the Lighting')

# Original vs uniform illuminated
# plt.figure()
# plt.subplot(1, 3, 1); plt.imshow(g, cmap='gray');     plt.title('Original');  plt.axis('off')
# plt.subplot(1, 3, 2); plt.imshow(light, cmap='gray'); plt.title('Lighting');  plt.axis('off')
# plt.subplot(1, 3, 3); plt.imshow(h3, cmap='gray');    plt.title('Uniformly Illuminated Image'); plt.axis('off')
# plt.show()

# %% Output
g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input Image', g)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% Image processing goes here
