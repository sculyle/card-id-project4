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
# %% Straighten the card (TODO)
# the card can be tilted on the desk, and a tilted card rotates every symbol (the A and the club on the ace picture)
# a rotated symbol will not match an upright template, so rotate the whole card upright once before the CCA
# idea: rotate the card mask over a range of angles (rotation matrix + interpolation from geometric transforms notes),
# the bounding box is smallest when the card is upright, so keep that angle and rotate the picture by it
# then the pips on the lower half are known to be upside down, and the corner symbols are known to be upright

# ---------------
# %% Connected component analysis (CCA notes)
# card is white on black, so invert: the dark symbols on the card (ranks, suits, pips) are the objects
# grow X_0 with X_k = (X_k-1 dilated by B) intersect A until X_k = X_k-1, that is one whole object
# then XOR it out of A and repeat to find the next object
B = np.ones((3, 3), np.uint8)              # B for 8-connected (default in notes), cross = 4-connected
MIN_PIXELS = 10                            # not from notes, my cutoff: smaller objects are leftover noise (specks were 2-4 px, symbols 160+ px)
A = (~b2).astype(np.uint8)
objects = []
while A.any():
    X = np.zeros_like(A)
    X[tuple(np.argwhere(A)[0])] = 1        # X_0: one point in an object
    while True:
        Xn = cv2.dilate(X, B) & A          # dilate, then intersect with A so it cant grow past the object
        if (Xn == X).all():                # same twice in a row, whole object found
            break
        X = Xn
    # keep it only if it is big enough, and it doesnt touch the image border (that is the black background, not a symbol)
    if X.sum() >= MIN_PIXELS and not (X[[0, -1]].any() or X[:, [0, -1]].any()):
        objects.append(X)
    A = A ^ X                              # remove the found object (XOR), A is empty when all are found
print('%d objects found' % len(objects))

# ---------------
# %% Objects output (one figure per object, commented out because it opens too many windows)
# # one figure per object found
# for k, X in enumerate(objects, 1):
#     plt.figure(); plt.imshow(X, cmap='gray'); plt.title('Object %d  (%d pixels)' % (k, X.sum())); plt.axis('off')
# plt.show()

# ---------------
# %% Fixed size symbols (experiment for matching, comment this whole section out if it doesnt work)
# bounding box = smallest and largest row/col of the object's points (not from notes)
# move the box corner to the origin (translation notes), then resize every object to the same size (bicubic, geometric transforms notes)
SIZE = 64
symbols = []
for X in objects:
    r, c = np.argwhere(X).T
    box = X[r.min():r.max() + 1, c.min():c.max() + 1]
    symbols.append(cv2.resize(box * 255, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC))

# ---------------
# %% Fixed size symbols output (comment this whole block out for the final project)
plt.figure(figsize=(12, 6))
for k, sym in enumerate(symbols, 1):
    plt.subplot(2, (len(symbols) + 1) // 2, k); plt.imshow(sym, cmap='gray'); plt.title('Object %d' % k); plt.axis('off')
plt.show()

# ---------------
# %% Skeleton fit (experiment, comment this whole section out if it doesnt work)
# skeletons drawn by hand with draw_skeleton.py, hard coded for now, each is (row, col) points on the 64 x 64 symbol
skel_club = np.array([   # 245 points
    (13, 31), (13, 32), (14, 31), (14, 32), (15, 31), (15, 32), (16, 31), (16, 32), (16, 33),
    (17, 29), (17, 30), (17, 31), (17, 32), (17, 33), (17, 34), (17, 35), (18, 28), (18, 29),
    (18, 30), (18, 31), (18, 32), (18, 33), (18, 34), (18, 35), (18, 36), (19, 28), (19, 29),
    (19, 30), (19, 31), (19, 32), (19, 33), (19, 34), (19, 35), (19, 36), (20, 28), (20, 29),
    (20, 30), (20, 31), (20, 32), (20, 33), (20, 34), (20, 35), (20, 36), (21, 28), (21, 29),
    (21, 30), (21, 31), (21, 32), (21, 34), (21, 35), (21, 36), (22, 28), (22, 29), (22, 30),
    (22, 31), (22, 32), (22, 33), (22, 34), (22, 35), (23, 28), (23, 29), (23, 30), (23, 31),
    (23, 32), (23, 33), (23, 34), (23, 35), (24, 28), (24, 29), (24, 30), (24, 31), (24, 32),
    (24, 33), (24, 34), (25, 30), (25, 31), (25, 32), (25, 33), (26, 31), (26, 32), (27, 31),
    (27, 32), (28, 31), (28, 32), (29, 31), (29, 32), (30, 31), (30, 32), (31, 31), (31, 32),
    (32, 30), (32, 31), (32, 32), (32, 33), (33, 29), (33, 30), (33, 31), (33, 32), (33, 33),
    (33, 34), (33, 35), (33, 40), (34, 27), (34, 28), (34, 29), (34, 34), (34, 35), (34, 36),
    (34, 39), (34, 40), (34, 41), (34, 42), (34, 43), (34, 44), (35, 21), (35, 22), (35, 23),
    (35, 24), (35, 26), (35, 27), (35, 28), (35, 36), (35, 37), (35, 38), (35, 39), (35, 40),
    (35, 41), (35, 42), (35, 43), (35, 44), (35, 45), (36, 20), (36, 21), (36, 22), (36, 23),
    (36, 24), (36, 25), (36, 26), (36, 37), (36, 38), (36, 39), (36, 40), (36, 41), (36, 42),
    (36, 43), (36, 44), (36, 45), (37, 20), (37, 21), (37, 22), (37, 23), (37, 24), (37, 25),
    (37, 26), (37, 27), (37, 38), (37, 39), (37, 40), (37, 41), (37, 42), (37, 43), (37, 44),
    (37, 45), (38, 20), (38, 21), (38, 22), (38, 23), (38, 24), (38, 25), (38, 26), (38, 27),
    (38, 37), (38, 38), (38, 39), (38, 40), (38, 41), (38, 42), (38, 43), (38, 44), (38, 45),
    (39, 20), (39, 21), (39, 22), (39, 23), (39, 25), (39, 26), (39, 27), (39, 37), (39, 38),
    (39, 39), (39, 40), (39, 41), (39, 42), (39, 43), (39, 44), (40, 20), (40, 21), (40, 22),
    (40, 23), (40, 24), (40, 25), (40, 26), (40, 27), (40, 37), (40, 38), (40, 39), (40, 40),
    (40, 41), (40, 42), (40, 43), (40, 44), (40, 45), (41, 19), (41, 20), (41, 21), (41, 22),
    (41, 23), (41, 24), (41, 25), (41, 26), (41, 39), (41, 40), (41, 41), (41, 42), (41, 44),
    (41, 45), (41, 46), (41, 47), (42, 18), (42, 19), (42, 20), (42, 46), (42, 47), (42, 48),
    (42, 49), (43, 17), (43, 18), (43, 19), (43, 48), (43, 49), (44, 16), (44, 17), (44, 18),
    (45, 16), (45, 17),
])

skel_spade = np.array([   # 327 points
    (13, 31), (13, 32), (14, 31), (14, 32), (15, 31), (15, 32), (15, 33), (16, 31), (16, 32),
    (16, 33), (17, 31), (17, 32), (17, 33), (17, 34), (18, 31), (18, 32), (18, 33), (18, 34),
    (19, 31), (19, 32), (19, 33), (19, 34), (20, 30), (20, 31), (20, 32), (20, 33), (20, 34),
    (21, 30), (21, 31), (21, 32), (21, 33), (21, 34), (22, 29), (22, 30), (22, 32), (22, 33),
    (22, 34), (23, 29), (23, 30), (23, 33), (23, 34), (23, 35), (24, 28), (24, 29), (24, 33),
    (24, 34), (24, 35), (25, 28), (25, 29), (25, 34), (25, 35), (25, 36), (25, 40), (26, 24),
    (26, 27), (26, 28), (26, 34), (26, 35), (26, 36), (26, 39), (26, 40), (27, 24), (27, 25),
    (27, 26), (27, 27), (27, 28), (27, 35), (27, 36), (27, 37), (27, 38), (27, 39), (27, 40),
    (28, 25), (28, 26), (28, 27), (28, 28), (28, 35), (28, 36), (28, 37), (28, 38), (28, 39),
    (29, 26), (29, 27), (29, 28), (29, 29), (29, 30), (29, 33), (29, 34), (29, 35), (29, 36),
    (29, 37), (29, 38), (30, 25), (30, 26), (30, 27), (30, 28), (30, 29), (30, 30), (30, 31),
    (30, 32), (30, 33), (30, 34), (30, 35), (30, 36), (30, 37), (30, 38), (31, 25), (31, 26),
    (31, 37), (31, 38), (31, 39), (32, 24), (32, 25), (32, 38), (32, 39), (32, 40), (33, 24),
    (33, 25), (33, 38), (33, 39), (33, 40), (34, 23), (34, 24), (34, 39), (34, 40), (34, 41),
    (35, 22), (35, 23), (35, 31), (35, 32), (35, 33), (35, 37), (35, 40), (35, 41), (36, 22),
    (36, 23), (36, 28), (36, 29), (36, 30), (36, 31), (36, 32), (36, 33), (36, 34), (36, 35),
    (36, 36), (36, 37), (36, 38), (36, 40), (36, 41), (36, 42), (37, 22), (37, 26), (37, 27),
    (37, 28), (37, 29), (37, 35), (37, 36), (37, 37), (37, 38), (37, 41), (37, 42), (37, 43),
    (37, 44), (38, 21), (38, 22), (38, 25), (38, 26), (38, 27), (38, 37), (38, 38), (38, 39),
    (38, 41), (38, 42), (38, 43), (38, 44), (38, 45), (39, 21), (39, 22), (39, 24), (39, 25),
    (39, 38), (39, 39), (39, 40), (39, 41), (39, 42), (39, 43), (39, 44), (39, 45), (39, 46),
    (40, 18), (40, 19), (40, 20), (40, 21), (40, 22), (40, 23), (40, 24), (40, 38), (40, 39),
    (40, 40), (40, 41), (40, 42), (40, 43), (40, 44), (40, 45), (40, 46), (40, 47), (41, 16),
    (41, 17), (41, 18), (41, 19), (41, 20), (41, 21), (41, 22), (41, 23), (41, 24), (41, 38),
    (41, 39), (41, 40), (41, 41), (41, 42), (41, 43), (41, 44), (41, 45), (41, 46), (41, 47),
    (42, 15), (42, 16), (42, 17), (42, 18), (42, 19), (42, 20), (42, 22), (42, 23), (42, 39),
    (42, 40), (42, 41), (42, 42), (42, 43), (42, 44), (42, 45), (42, 46), (42, 47), (43, 15),
    (43, 16), (43, 17), (43, 18), (43, 19), (43, 20), (43, 21), (43, 22), (43, 23), (43, 24),
    (43, 39), (43, 40), (43, 41), (43, 42), (43, 43), (43, 44), (43, 45), (43, 46), (43, 47),
    (44, 15), (44, 17), (44, 18), (44, 19), (44, 20), (44, 21), (44, 22), (44, 23), (44, 24),
    (44, 40), (44, 41), (44, 42), (44, 43), (44, 44), (44, 45), (44, 46), (44, 47), (45, 15),
    (45, 16), (45, 17), (45, 18), (45, 19), (45, 20), (45, 21), (45, 22), (45, 23), (45, 24),
    (45, 41), (45, 42), (45, 43), (45, 44), (45, 45), (45, 46), (45, 47), (46, 15), (46, 16),
    (46, 17), (46, 18), (46, 19), (46, 20), (46, 23), (46, 24), (46, 42), (46, 43), (46, 44),
    (46, 45), (46, 46), (46, 47), (47, 15), (47, 16), (47, 17), (47, 18), (47, 19), (47, 20),
    (47, 22), (47, 23), (48, 16), (48, 17), (48, 18), (48, 19), (48, 20), (48, 21), (48, 22),
    (48, 23), (49, 19), (49, 20),
])

skel_heart = np.array([   # 287 points
    (17, 36), (17, 37), (18, 23), (18, 24), (18, 25), (18, 26), (18, 27), (18, 28), (18, 33),
    (18, 34), (18, 35), (18, 36), (18, 37), (18, 38), (18, 39), (18, 40), (19, 22), (19, 23),
    (19, 27), (19, 28), (19, 29), (19, 32), (19, 33), (19, 34), (19, 36), (19, 37), (19, 39),
    (19, 40), (19, 41), (20, 21), (20, 22), (20, 29), (20, 30), (20, 31), (20, 32), (20, 35),
    (20, 36), (20, 37), (20, 38), (20, 39), (20, 41), (20, 42), (21, 20), (21, 21), (21, 26),
    (21, 27), (21, 30), (21, 31), (21, 32), (21, 34), (21, 35), (21, 36), (21, 37), (21, 39),
    (21, 41), (21, 42), (22, 20), (22, 21), (22, 25), (22, 26), (22, 27), (22, 28), (22, 29),
    (22, 30), (22, 31), (22, 34), (22, 35), (22, 36), (22, 37), (22, 38), (22, 39), (22, 42),
    (22, 43), (23, 20), (23, 25), (23, 26), (23, 27), (23, 28), (23, 29), (23, 30), (23, 31),
    (23, 33), (23, 34), (23, 35), (23, 37), (23, 38), (23, 39), (23, 42), (23, 43), (24, 19),
    (24, 20), (24, 24), (24, 25), (24, 26), (24, 28), (24, 29), (24, 30), (24, 31), (24, 33),
    (24, 34), (24, 35), (24, 38), (24, 39), (24, 42), (24, 43), (25, 19), (25, 20), (25, 24),
    (25, 25), (25, 26), (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34),
    (25, 35), (25, 37), (25, 38), (25, 39), (25, 42), (25, 43), (26, 19), (26, 20), (26, 23),
    (26, 24), (26, 25), (26, 26), (26, 27), (26, 28), (26, 29), (26, 30), (26, 31), (26, 32),
    (26, 33), (26, 34), (26, 35), (26, 36), (26, 37), (26, 38), (26, 42), (26, 43), (27, 19),
    (27, 20), (27, 24), (27, 25), (27, 26), (27, 27), (27, 28), (27, 29), (27, 30), (27, 32),
    (27, 33), (27, 34), (27, 35), (27, 36), (27, 38), (27, 42), (27, 43), (28, 20), (28, 24),
    (28, 25), (28, 26), (28, 27), (28, 28), (28, 32), (28, 33), (28, 34), (28, 38), (28, 41),
    (28, 42), (29, 20), (29, 21), (29, 25), (29, 26), (29, 27), (29, 28), (29, 29), (29, 30),
    (29, 31), (29, 32), (29, 33), (29, 34), (29, 35), (29, 37), (29, 38), (29, 40), (29, 41),
    (29, 42), (30, 20), (30, 21), (30, 26), (30, 27), (30, 28), (30, 29), (30, 30), (30, 31),
    (30, 32), (30, 33), (30, 34), (30, 35), (30, 36), (30, 37), (30, 38), (30, 39), (30, 40),
    (30, 41), (31, 21), (31, 22), (31, 27), (31, 28), (31, 29), (31, 30), (31, 31), (31, 32),
    (31, 33), (31, 34), (31, 35), (31, 36), (31, 37), (31, 38), (31, 39), (31, 40), (32, 22),
    (32, 23), (32, 27), (32, 28), (32, 29), (32, 30), (32, 31), (32, 32), (32, 33), (32, 34),
    (32, 35), (32, 36), (32, 37), (32, 38), (33, 23), (33, 24), (33, 29), (33, 30), (33, 31),
    (33, 32), (33, 35), (33, 36), (33, 37), (34, 23), (34, 24), (34, 25), (34, 26), (34, 34),
    (34, 35), (34, 36), (35, 25), (35, 26), (35, 27), (35, 33), (35, 34), (35, 35), (36, 27),
    (36, 28), (36, 33), (36, 34), (37, 28), (37, 29), (37, 32), (37, 33), (38, 29), (38, 30),
    (38, 32), (38, 33), (39, 29), (39, 30), (39, 32), (40, 30), (40, 31), (40, 32), (41, 31),
    (41, 32), (42, 31), (42, 32), (43, 31), (43, 32), (44, 31), (44, 32), (45, 31),
])

skel_diamond = np.array([   # 328 points
    (13, 32), (14, 31), (14, 32), (14, 33), (15, 30), (15, 31), (15, 32), (15, 33), (15, 34),
    (16, 29), (16, 30), (16, 31), (16, 32), (16, 34), (16, 35), (17, 28), (17, 29), (17, 32),
    (17, 33), (17, 35), (17, 36), (18, 28), (18, 29), (18, 31), (18, 32), (18, 35), (18, 36),
    (19, 27), (19, 28), (19, 30), (19, 31), (19, 32), (19, 33), (19, 34), (19, 35), (19, 36),
    (19, 37), (20, 26), (20, 27), (20, 34), (20, 35), (20, 37), (20, 38), (21, 26), (21, 27),
    (21, 33), (21, 34), (21, 37), (21, 38), (21, 39), (22, 25), (22, 26), (22, 30), (22, 31),
    (22, 32), (22, 33), (22, 38), (22, 39), (23, 24), (23, 25), (23, 29), (23, 30), (23, 31),
    (23, 39), (23, 40), (24, 24), (24, 25), (24, 28), (24, 29), (24, 33), (24, 34), (24, 35),
    (24, 36), (24, 37), (24, 38), (24, 39), (24, 40), (24, 41), (25, 23), (25, 24), (25, 28),
    (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37),
    (25, 38), (25, 39), (25, 40), (25, 41), (25, 42), (26, 22), (26, 23), (26, 26), (26, 27),
    (26, 28), (26, 29), (26, 30), (26, 31), (26, 32), (26, 33), (26, 34), (26, 35), (26, 36),
    (26, 37), (26, 38), (26, 39), (26, 41), (26, 42), (27, 22), (27, 23), (27, 24), (27, 25),
    (27, 26), (27, 27), (27, 28), (27, 29), (27, 30), (27, 31), (27, 32), (27, 33), (27, 42),
    (27, 43), (28, 21), (28, 22), (28, 23), (28, 36), (28, 37), (28, 38), (28, 39), (28, 43),
    (28, 44), (29, 20), (29, 21), (29, 22), (29, 23), (29, 24), (29, 25), (29, 26), (29, 27),
    (29, 28), (29, 29), (29, 30), (29, 31), (29, 32), (29, 33), (29, 34), (29, 35), (29, 36),
    (29, 37), (29, 38), (29, 39), (29, 44), (29, 45), (30, 19), (30, 20), (30, 36), (30, 37),
    (30, 38), (30, 39), (30, 45), (30, 46), (31, 18), (31, 19), (31, 27), (31, 28), (31, 29),
    (31, 30), (31, 31), (31, 32), (31, 33), (31, 34), (31, 35), (31, 36), (31, 37), (31, 38),
    (31, 39), (31, 40), (31, 45), (31, 46), (32, 17), (32, 18), (32, 24), (32, 25), (32, 26),
    (32, 27), (32, 28), (32, 29), (32, 30), (32, 31), (32, 32), (32, 33), (32, 34), (32, 35),
    (32, 36), (32, 37), (32, 38), (32, 39), (32, 46), (33, 16), (33, 17), (33, 22), (33, 23),
    (33, 24), (33, 25), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30), (33, 31), (33, 32),
    (33, 33), (33, 34), (33, 35), (33, 37), (33, 38), (33, 39), (33, 45), (33, 46), (34, 16),
    (34, 17), (34, 18), (34, 19), (34, 20), (34, 22), (34, 23), (34, 24), (34, 25), (34, 26),
    (34, 27), (34, 28), (34, 29), (34, 30), (34, 31), (34, 36), (34, 37), (34, 43), (34, 44),
    (34, 45), (35, 19), (35, 20), (35, 21), (35, 34), (35, 35), (35, 36), (35, 41), (35, 42),
    (35, 43), (36, 21), (36, 22), (36, 23), (36, 33), (36, 34), (36, 35), (36, 39), (36, 40),
    (36, 41), (36, 42), (37, 22), (37, 23), (37, 24), (37, 32), (37, 33), (37, 38), (37, 39),
    (37, 40), (38, 24), (38, 25), (38, 32), (38, 33), (38, 36), (38, 37), (38, 38), (39, 25),
    (39, 26), (39, 31), (39, 32), (39, 35), (39, 36), (39, 37), (40, 25), (40, 26), (40, 31),
    (40, 34), (40, 35), (40, 36), (40, 37), (41, 26), (41, 27), (41, 31), (41, 33), (41, 34),
    (41, 35), (41, 36), (42, 27), (42, 28), (42, 30), (42, 31), (42, 32), (42, 33), (42, 35),
    (42, 36), (43, 28), (43, 29), (43, 31), (43, 34), (43, 35), (44, 28), (44, 29), (44, 34),
    (44, 35), (45, 29), (45, 30), (45, 33), (45, 34), (46, 30), (46, 31), (46, 32), (46, 33),
    (47, 31), (47, 32), (47, 33), (48, 32),
])

skel_raw = {'club': skel_club, 'spade': skel_spade, 'heart': skel_heart, 'diamond': skel_diamond}
# NOTE: these skeletons will need updating whenever we find a symbol that gets misclassified
# (for example the spade skeleton fits inside the clubs on the ace), redraw it and paste the new points in above

# --- DRAW YOUR OWN SKELETON (optional, eventually comment this whole part out) ---
DRAW_NAME = ''     # 'club', 'spade', 'heart' or 'diamond' = draw that one yourself, '' = use the hard coded ones
if DRAW_NAME:
    # hold left mouse button to draw an upright skeleton, c = clear, s = done, the symbols are stretched to fill
    # the 64 x 64 box so it gets scaled up like the hard coded ones below
    BIG = 512                              # drawing size, shrunk to 64 x 64 after
    canvas = np.zeros((BIG, BIG), np.uint8)
    last = None

    def mouse(event, x, y, flags, param):
        global last
        if event == cv2.EVENT_LBUTTONDOWN:
            last = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and last is not None:
            cv2.line(canvas, last, (x, y), 255, 8)
            last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            last = None

    cv2.namedWindow('Draw skeleton')
    cv2.setMouseCallback('Draw skeleton', mouse)
    while True:
        show = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(show, (64, 64), (BIG - 64, BIG - 64), (0, 0, 255), 1)      # guide box
        cv2.line(show, (BIG // 2, 0), (BIG // 2, BIG), (0, 0, 120), 1)            # center line
        cv2.imshow('Draw skeleton', show)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            canvas[:] = 0
        if key == ord('s'):
            break
    cv2.destroyAllWindows()
    skel_raw[DRAW_NAME] = np.argwhere(cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 60)
    print('skel_%s = np.array([%s])' % (DRAW_NAME, ', '.join('(%d, %d)' % (r, c) for r, c in skel_raw[DRAW_NAME])))
# --- END DRAW YOUR OWN SKELETON ---

# the drawings are smaller than the symbols (which are stretched to fill the whole box), so scale them the same way:
# move the skeleton's box corner to the origin and stretch it to the box size, MARGIN px from the edge so it fits inside
MARGIN = 8                                 # picked by trial on my test cards
def fill_box(pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return np.round(MARGIN + (pts - lo) * (SIZE - 1 - 2 * MARGIN) / (hi - lo)).astype(int)

skeletons = {name: fill_box(pts) for name, pts in skel_raw.items()}

def rotated(pts, deg):
    # rotate the points about the middle (rotation matrix from geometric transforms notes), as a 64 x 64 mask
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    r, c = np.round((pts - 32) @ R.T + 32).astype(int).T
    m = np.zeros((SIZE, SIZE), np.uint8)
    m[r.clip(0, SIZE - 1), c.clip(0, SIZE - 1)] = 1
    return m

# erosion notes: A erode B is every place B fits entirely inside A, so any left over = the skeleton fits
# the card can be tilted or upside down, so rotate the skeleton and try every angle
# try every skeleton on every object, keep the names of the ones that fit
labels = []
for sym in symbols:
    A = (sym > 127).astype(np.uint8)
    names = []
    for name, skel in skeletons.items():
        # symbols and skeletons are the same size and centered, so only rotate it, look at the middle only (no sliding around)
        if any(cv2.erode(A, rotated(skel, deg), borderValue=0)[SIZE // 2, SIZE // 2]
               for deg in range(0, 360, 5)):
            names.append(name)
    labels.append(names)

# ---------------
# %% Skeleton fit output (comment this whole block out for the final project)
# write what each object IS on the card, objects that fit nothing get no label
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); plt.axis('off')
for X, names in zip(objects, labels):
    if names:
        y, x = np.argwhere(X).mean(axis=0)                 # middle of the object on the card
        plt.text(x, y, ', '.join(names), color='lime', ha='center', va='center', fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, pad=1))
plt.show()

# ---------------
# %% Image processing goes here
