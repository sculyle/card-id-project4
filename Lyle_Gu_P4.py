# clear; close all; clc;
import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # work from the script's folder

# ---------------
# %% Settings
ZOOM_PCT = 70        # 1.7 zoom
CAM_W, CAM_H = 1280, 720   # webcam resolution, default is 640 x 480
AVG_FRAMES = 10      # number of frames averaged for the picture
saveDir = 'captures'

def zoom(img):
    # Just artifically zooming in the webcam on the setup, not actual detection processing yet
    # basically crops to center
    h, w = img.shape[:2]
    s = 1 + ZOOM_PCT / 100
    ch, cw = round(h / s), round(w / s)
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC)

# ---------------
print('1. Acquire from Webcam')
print('2. Read from File')
choice = input('Choose option:  ').strip()

if choice == '1':
    # spacebar = take picture, q = quit
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit('Could not open camera')
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # fixed focus
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)    # more pixels on the card
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    while True:
        # asked ai to help me create settings for capture from webcam
        ok, f = cap.read()
        f = zoom(f)
        cv2.putText(f, 'SPACE = take picture   q = quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Preview', f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            raise SystemExit('No picture taken')
    for _ in range(60):                     # let auto exposure settle
        ok, f = cap.read()

    # ---------------
    # Frame averaging (restoration notes): g_bar = (1/M) * sum of g_i
    # the card and camera stay still so only the noise changes and it cancels out
    total = np.zeros(f.shape)
    for _ in range(AVG_FRAMES):
        ok, f = cap.read()
        total += f
    f = zoom(np.round(total / AVG_FRAMES).astype(np.uint8))
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

# ---------------
# %% Grayscale
g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

# ---------------
# %% Even lighting -- didnt really work how i thought it would the binarization works better
# low-pass to estimate the lighting, subtract it (same as P3, Butterworth from 2D filtering notes)
# F = np.fft.fftshift(np.fft.fft2(g.astype(float)))
# M, N = F.shape
# y, x = np.indices(F.shape)
# D = np.hypot(x - N // 2, y - M // 2)         # distance from center
# HL = 1 / (1 + (D / 10)**4)                   # order n = 2, cutoff D0 = 10
# light = np.real(np.fft.ifft2(np.fft.ifftshift(F * HL)))
# h3 = g - light
#
# plt.figure(); plt.imshow(HL, extent=[-N//2, N//2, M//2, -M//2], cmap='gray')
# plt.xlim(-50, 50); plt.ylim(50, -50); plt.title('Filter Used')
# plt.figure(); plt.imshow(h3, cmap='gray'); plt.title('Uniformly Illuminated Image')
# plt.show()

# ---------------
# %% Noise filters - not really having a huge affect but feel free to test out
# mean, median, and adaptive (Lee) from the image restoration notes
# gf = g.astype(float)
#
# def lee(k):    # f_hat = g - (var_n / var_L) * (g - mean_L)
#     m = cv2.blur(gf, (k, k))
#     v = cv2.blur(gf**2, (k, k)) - m**2
#     vn = np.median(v)    # noise variance, most of the picture is flat so typical local variance is noise
#     return gf - vn / np.maximum(v, vn + 1e-6) * (gf - m)
#
# filters = {'Mean': lambda k: cv2.blur(g, (k, k)),
#            'Median': lambda k: cv2.medianBlur(g, k),
#            'Adaptive': lee}
# for name, fn in filters.items():
#     plt.figure(figsize=(11, 8))
#     for i, (t, im) in enumerate([('Original', g)] + [('%s %d x %d' % (name, k, k), fn(k)) for k in [3, 5, 9]]):
#         plt.subplot(2, 2, i + 1); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
# plt.show()

# ---------------

# %% Binarize
# median filter worked the best
# removes speckle without blurring edges like a mean filter
gm = cv2.medianBlur(g, 3)

# frome notes: s = 255/(rmax - rmin) * (r - rmin)
linear = []
for im in (g, gm):
    rmin = int(im.min())
    rmax = int(im.max())
    scale = 255 / (rmax - rmin)
    s = np.zeros_like(im)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            s[y, x] = round(scale * (int(im[y, x]) - rmin))
    linear.append(s)
gl, gml = linear

# threshold to a binary image, T = 128 picked by trial on my test cards
T = 128
b1 = gl > T
b2 = gml > T

# ---------------
# %% Histograms (comment this whole block out for the final project)
plt.figure(figsize=(14, 4))
for i, (t, im) in enumerate([('Original', g), ('Linear', gl),
                             ('Median + linear', gml), ('Binarized', b2 * 255)]):
    plt.subplot(1, 4, i + 1); plt.hist(im.ravel(), 256, (0, 256)); plt.title(t)
plt.show()

# ---------------
# %% Output (comment this whole block out for the final project)
plt.figure(figsize=(14, 5))
for i, (t, im) in enumerate([('Original', gl), ('Threshold, no filter', b1),
                             ('Threshold, 3 x 3 median first', b2)]):
    plt.subplot(1, 3, i + 1); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
plt.show()

# ---------------
# %% Connected component analysis (X_k = (X_k-1 dilated by B) intersect A, until X_k = X_k-1)
B = np.ones((3, 3), np.uint8)              # 8-connected (default in notes)
A = (~b2).astype(np.uint8)                 # invert, the dark symbols on the card are the objects
objects = []
while A.any():
    X = np.zeros_like(A)
    y, x = np.argwhere(A)[0]               # X_0: one point in an object
    X[y, x] = 1
    while True:
        Xn = cv2.dilate(X, B) & A          # grow, but A limits it to the object
        if (Xn == X).all():                # same twice in a row, whole object found
            break
        X = Xn
    if not (X[0].any() or X[-1].any() or X[:, 0].any() or X[:, -1].any()):
        objects.append(X)                  # skip the black background, it touches the image border
    A = A ^ X                              # remove the found object (XOR), then look for the next
print('%d objects found' % len(objects))

# ---------------
# %% Objects output (comment this whole block out for the final project)
for k, X in enumerate(objects, 1):
    plt.figure(); plt.imshow(X, cmap='gray'); plt.title('Object %d  (%d pixels)' % (k, X.sum())); plt.axis('off')
plt.show()

# ---------------
# %% Image processing goes here
