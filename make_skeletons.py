# side tool, not part of the project script: make skeletons from real objects found on a card picture
# finds the objects the same way as Lyle_Gu_P4.py, shows them numbered, then you pick which ones are good
# a skeleton here = the object shrunk by erosion (erosion notes), so it fits inside that kind of symbol like the real thing
# prints skel_<name> = np.array([...]) and adds it to skeletons_from_objects.txt, paste into Lyle_Gu_P4.py
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # work from the script's folder

SIZE = 64            # same as the symbols in Lyle_Gu_P4.py
T = 128              # same threshold
MIN_PIXELS = 10      # same noise cutoff
saveDir = 'captures'

# ---------------
# %% Read from file
names = [n for e in ('png', 'jpg', 'jpeg', 'tif', 'tiff')
         for n in glob.glob(saveDir + '/*.' + e) + glob.glob('*.' + e)]
if not names:
    raise SystemExit('No image found')
for k, n in enumerate(names, 1):
    print('%d: %s' % (k, n))
f = cv2.imread(names[int(input('Choose image from above:  ')) - 1])

# ---------------
# %% Same steps as Lyle_Gu_P4.py: gray, 3 x 3 median, linear stretch, threshold, invert
g = cv2.medianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 3)
gl = 255 / (int(g.max()) - int(g.min())) * (g.astype(float) - int(g.min()))     # s = 255/(rmax - rmin) * (r - rmin)
A = (~(gl > T)).astype(np.uint8)

# ---------------
# %% Connected component analysis (grow, intersect with A, until same twice, XOR out)
B = np.ones((3, 3), np.uint8)
objects = []
while A.any():
    X = np.zeros_like(A)
    X[tuple(np.argwhere(A)[0])] = 1
    while True:
        Xn = cv2.dilate(X, B) & A
        if (Xn == X).all():
            break
        X = Xn
    if X.sum() >= MIN_PIXELS and not (X[[0, -1]].any() or X[:, [0, -1]].any()):
        objects.append(X)
    A = A ^ X

# ---------------
# %% Fixed size symbols (bounding box, resize to SIZE x SIZE)
symbols = []
for X in objects:
    r, c = np.argwhere(X).T
    box = X[r.min():r.max() + 1, c.min():c.max() + 1]
    symbols.append((cv2.resize(box * 255, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) > 127).astype(np.uint8))

plt.figure(figsize=(12, 6))
for k, sym in enumerate(symbols, 1):
    plt.subplot(2, (len(symbols) + 1) // 2, k); plt.imshow(sym, cmap='gray'); plt.title('Object %d' % k); plt.axis('off')
plt.show()

# ---------------
# %% Make skeletons from the objects you pick
while True:
    k = input('Object number to make a skeleton from (blank = done):  ').strip()
    if not k:
        break
    name = input('What is it (club, spade, heart, diamond):  ').strip()
    n = int(input('How many erosions (blank = 3):  ') or 3)
    skel = symbols[int(k) - 1]
    for _ in range(n):
        skel = cv2.erode(skel, B, borderValue=0)                          # shrink it, so the real symbol fits around it
    pts = np.argwhere(skel)
    text = 'skel_%s = np.array([%s])   # (row, col), %d points, from object %s, %d erosions\n' % (
        name, ', '.join('(%d, %d)' % (r, c) for r, c in pts), len(pts), k, n)
    print(text)
    with open('skeletons_from_objects.txt', 'a') as fh:
        fh.write(text)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(symbols[int(k) - 1], cmap='gray'); plt.title('Object %s' % k); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(skel, cmap='gray'); plt.title('Skeleton: %s' % name); plt.axis('off')
    plt.show()
