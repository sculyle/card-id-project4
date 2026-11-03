"""
ECE 4367 / 5367  Image Processing  -  Project 4   (version 5: no templates)
Lyle & Gu:  determine the rank and suit of a playing card

Usage
    python Lyle_Gu_P4.py
    1 = Acquire from Webcam
          live preview;  SPACE = take a picture and recognize it,  q = quit.
          Close the result window to go back to the preview for the next card.
          Every picture is also saved in ./captures.
    2 = Read from File
          type the full path of an image including the file name, e.g. C:\\Billy\\card.jpg
          (or press Enter to choose from the card_* images in this folder and in ./captures);
          q = quit.  Close the result window to read the next image.
    For every card a window opens with the input picture, the straightened and cropped card
    (counted pips in green boxes) and the corner index, and the result is printed in the
    console, e.g.  "Result: Queen of Spades"  plus what it was based on.

Files (all in this folder)
    Lyle_Gu_P4.py      the program; it needs nothing else (no templates, no training data)
    capture_cards.py   side tool: take labeled test pictures with the same camera settings
    evaluate.py        side tool: run the program on a folder of labeled pictures, print accuracy
    Needs Python with numpy, opencv-python, matplotlib.

Pipeline (the card is treated as grayscale, no colour is used)
    1. Gray, 3 x 3 median filter, uneven lighting removed: the background is estimated by a
       gray-level opening with a square bigger than the card (the card and its symbols do not
       fit, so only the slowly changing background is left) and subtracted (top-hat).
    2. Tilt angle: smoothed image -> Sobel gx, gy -> gradient direction folded into [0, 90) deg
       -> histogram weighted by gradient magnitude (0.5 deg bins) -> peak, refined by the
       weighted mean of the neighbouring bins.  The four card edges are the longest straight
       edges, so they make the peak.
    3. Card: rotate by the angle (new area filled with black), threshold halfway between the
       darkest and the brightest pixel, count the white pixels in every column and row; the
       card is where the count is above half of the maximum.
       Crop, turn to portrait, resize to 250 x 350 (and a 4 x bigger copy for the corner index, CHANGED v5).
    4. Symbols on the card: threshold halfway between darkest and brightest, connected component
       analysis (dilate-intersect-XOR from the notes); the thin frame lines of the face pictures
       are removed first (CHANGED by Lyle); drop objects touching the card border
       and the two corner indexes (found in step 6).
    5. Face card or number card  (CHANGED in version 5, see "CHANGED (v5)" in the code)
       the picture of a J, Q, K is drawn inside a rectangular frame.  Threshold the card a little
       lighter than halfway (60 % of the way from darkest to brightest: a thin line is blurred to
       gray), widen the dark pixels sideways by LINE_TOL px (a slightly slanted line stays one run)
       and find the longest dark run down any column (the outer 6 % of the card is not searched).
       Longer than half of the card height -> face card.  On our photos the frame is at least 0.83
       of the height (0.62 on a deck with small pictures); on number cards and aces the longest run
       is one pip or the big ace pip (at most 0.41).
       Some decks print a box around the pips of the number cards, which is also a long line.  So if
       the objects inside (step 4) are 2 or more pips that all have the shape of the index suit symbol
       (filled, scaled into the same square, overlap IoU >= 0.70, either way up), it is a number card
       with a box.  A face picture never breaks into such pieces (on our cards at most 0.26).
       It does not need the picture to be one object (Gu's first rule, which
       failed when the picture broke into pieces), and it does not depend on how busy the middle
       of the card is (Lyle's rule: black / white switches down the middle fifth of the columns,
       more than 9.5 = face; the Bicycle ace of spades has 18 and was read as a Queen).
       Both earlier rules are kept as comments.
    6. Corner index  (CHANGED by Lyle, see "CHANGED" in the code)
       threshold the whole card, remove the long thin frame lines of the face pictures (white
       top-hat + opening with long line structuring elements), connected components, drop objects
       touching the card border or tiny.  The index is the leftmost object plus the objects
       stacked with it (rank above suit, "10" = two objects side by side); no fixed box, so a
       bigger or smaller index works.  Cut it out, enlarge 4 x, keep only those objects; top
       object = rank symbol, next = suit symbol.  If this fails the card is turned 180 deg and
       the bottom-right index is used.  The index (and its mirror in the other corner) is also
       left out of the pip count.
       Gu's original (a fixed top-left box of 21 % x 29 %) is only the last resort if the
       leftmost objects can not be split into a rank and a suit in either corner.
       CHANGED (v5): the index is cut from the 4 x copy of the card instead of enlarging the
       250 x 350 card, so the detail of the photo is kept (a card is about 580 px tall on our
       webcam pictures and 900 - 1000 px on the phone photos).
    7. Suit from the row-width profile of a suit symbol, in this order  (order CHANGED in v5):
         top/bottom mirror symmetric                 -> diamond (the only suit that looks the same
                                                        upside down; checked first because it is the
                                                        clearest feature and blur does not change it)
         top 15 % of the rows wide (two lobes)      -> heart
         top 15 % medium (one round lobe)            -> club,  otherwise (pointed top) spade
       CHANGED (v5): on a number card (2 - 10) the symbols are the pips, 2 - 3 x bigger than the
       index symbol, so their shape survives blur better: pips in the lower half are printed upside
       down and are turned back, the median of the features of all pips is used.  Face cards and
       aces use the index (the ace's single middle pip can be either way up).
    8. Rank
         number card: count the pips (objects at least 40 % of the biggest one and not a line, i.e. at
                      most 3 x longer than wide): 1 = A, 2 - 10
         face card:   look at the index letter:  a closed hole -> Q;
                      otherwise the fullest column (the long vertical stroke) on the left -> K,
                      on the right -> J

Assumptions (limits of the method)
    - One card per picture, face up, fully inside the picture; no joker.
    - Standard design deck: the rank and suit are printed clearly in the top-left and bottom-right
      corners and are separate from the picture and the pips (not touching them); indices are
      A, 2-10, J, Q, K; pips do not touch each other; face cards have a large picture.
    - Plain background darker than the card; its brightness may change slowly, but no other
      bright objects or patterns.
    - The card's short side is less than half of the picture's short side, and the index is big
      enough to read (the smallest card we tested was about 230 px tall).
    - Card flat and the camera roughly facing it (only slight perspective); any in-plane rotation.
    - No strong glare or hard shadow on the index or the pips; nothing covering the card
      (if one index is covered, the other one is used).
    - Face cards have their picture inside a frame (a straight line longer than half of the card).
      Number cards and aces have no long straight line, or only a box with nothing but pips inside.
    - The thresholds (frame length, J/K stroke position, suit rules) were set on our photos: the
      QB/T 2228 deck (52), a Bicycle standard deck (52) and Lyle's webcam pictures (15).
    Colour is not used, so grayscale pictures work too.
"""
import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------
# %% Settings
ZOOM_PCT = 70                    # webcam: 1.7 x digital zoom (crop to the center)
CAM_W, CAM_H = 1280, 720         # webcam resolution, default is 640 x 480
AVG_FRAMES = 10                  # frames averaged for one picture
SAVE_DIR = os.path.join(HERE, 'captures')

CARD_W, CARD_H = 250, 350        # normalized card size
ANGLE_SIZE = 1000                # longest side of the picture used to find the angle
ANGLE_SIGMA = 3                  # gaussian smoothing before Sobel
BG_SE = 0.5                      # background opening: square side, fraction of the picture's short side
BG_SCALE = 8                     # the background is estimated on a 1/8 size copy (it changes slowly)
CORNER_W, CORNER_H = 0.21, 0.29  # typical corner index size, fraction of the card (sizes the filters and the noise limit)
UP = 4                           # index enlargement
MIN_AREA = 0.003                 # objects smaller than this fraction of a typical corner are noise
FRAME_DISK = 0.10                # top-hat disk, fraction of the corner width
FRAME_LINE = 0.40                # line length, fraction of the corner height
# CHANGED (Lyle): the index is found as the leftmost objects instead of a fixed corner box
INDEX_ROWS, INDEX_COLS = 0.50, 0.40   # the index is searched for in the top-left part of the card
INDEX_GAP = 0.012                # gap (fraction of the card width) allowed between the digits of a "10"
INDEX_PAD = 3                    # pixels around the index objects when they are cut out
# CHANGED (Lyle): thin frame lines (a box around the pips, the frame of the face pictures) are removed before the pips are counted
FRAME_K = int(round(FRAME_DISK * CORNER_W * CARD_W)) | 1
FRAME_L = int(round(FRAME_LINE * CORNER_H * CARD_H))

# Gu's original face rule (commented out, replaced by the BUSY_* rule below):
# FACE_H, FACE_W = 0.60, 0.50      # face picture: one object taller / wider than this part of the card
# CHANGED (Lyle): face card found from the middle of the card (commented out in v5, replaced by the FRAME_* rule below)
# BUSY_X, BUSY_Y = (0.40, 0.60), (0.10, 0.90)   # middle of the card: the columns and rows that are checked
# BUSY_MAX = 9.5                   # (was 11, CHANGED (Lyle): the unique Q of diamonds has 11) a number card switches black / white at most this often down its middle
# CHANGED (v5): face card found from the frame around the picture
FRAME_FACE = 0.50                # face card: a vertical line longer than half of the card height
FRAME_LEVEL = 0.60               # line threshold: 60 % of the way from the darkest to the brightest pixel
LINE_TOL = 15                    # px the line may wander sideways (left-over tilt, slight perspective)
EDGE = 0.06                      # the outer 6 % of the card is not searched (card edge, background left by the crop)
SHAPE_N = 32                     # shapes are compared at 32 x 32
PIP_SHAPE = 0.70                 # a pip overlaps the index suit symbol by at least 70 % (IoU)
PIP_LONG = 3                     # a pip is at most 3 x longer than wide (a line is not a pip)
PIP_UPRIGHT = 0.52               # pips with their middle above 52 % of the card height are upright, below upside down
PIP_REL = 0.40                   # pip: at least 40 % of the biggest object on the card ...
PIP_MIN = 0.003                  # ... and at least 0.3 % of the card
HOLE_MIN = 0.02                  # a hole in a letter is at least 2 % of the letter's box
STEM_K = 0.40                    # fullest column left of 40 % of the letter width -> K, else J

ASYM_DIAMOND = 0.115             # suit rules, checked in this order (CHANGED v5): diamonds <= 0.05, others >= 0.18
TOP_HEART = 0.47                 # hearts >= 0.59, clubs 0.29 - 0.36
TOP_CLUB = 0.26                  # CHANGED (v5, was 0.235): spades <= 0.20 on sharp photos, but blur
                                 # widens the pointed top, so the limit sits a little above the middle

RANK_NAMES = {'A': 'Ace', 'J': 'Jack', 'Q': 'Queen', 'K': 'King'}
SUIT_NAMES = {'c': 'Clubs', 'd': 'Diamonds', 'h': 'Hearts', 's': 'Spades'}


# ---------------
# %% Acquisition (webcam part from Lyle's version)
def zoom(img):
    # digital zoom: crop the center and resize back
    h, w = img.shape[:2]
    s = 1 + ZOOM_PCT / 100
    ch, cw = round(h / s), round(w / s)
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC)


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit('Could not open camera')
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # fixed focus
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    return cap


def capture(cap):
    # preview until SPACE (picture) or q (None)
    while True:
        ok, f = cap.read()
        f = zoom(f)
        cv2.putText(f, 'SPACE = take picture   q = quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Preview', f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            return None
    for _ in range(60):                          # let auto exposure settle
        cap.read()
    # frame averaging (restoration notes): g_bar = (1/M) * sum of g_i, only the noise changes
    total = None
    for _ in range(AVG_FRAMES):
        ok, f = cap.read()
        total = f.astype(np.float64) if total is None else total + f
    f = zoom(np.round(total / AVG_FRAMES).astype(np.uint8))
    os.makedirs(SAVE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(SAVE_DIR, time.strftime('card_%Y%m%d_%H%M%S.png')), f)
    return f


def ask(prompt):
    try:
        return input(prompt).strip().strip('"').strip("'")
    except EOFError:                              # no console input left -> quit
        return 'q'


def read_file():
    # ask for one image until it can be read; None = the user typed q
    while True:
        path = ask('\nFull path of the image, including the file name '
                   '(Enter = choose from the images in this folder, q = quit):  ')
        if path.lower() == 'q':
            return None
        if not path:
            names = [n for e in ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp')
                     for n in glob.glob(os.path.join(SAVE_DIR, '*.' + e)) + glob.glob(os.path.join(HERE, 'card_*.' + e))]
            if not names:
                print('No card_*.png / .jpg in this folder or in captures')
                continue
            for k, n in enumerate(names, 1):
                print('%d: %s' % (k, os.path.relpath(n, HERE)))
            pick = ask('Choose image from above (q = quit):  ')
            if pick.lower() == 'q':
                return None
            if not pick.isdigit() or not 1 <= int(pick) <= len(names):
                print('Not a number from the list')
                continue
            path = names[int(pick) - 1]
        f = cv2.imread(path)
        if f is None:
            print('Could not read ' + path)
            continue
        print('Reading ' + path)
        return f


# ---------------
# %% Step 1-3: find the card, straighten and crop it
def to_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.medianBlur(g, 3)                   # removes speckle without blurring edges


def remove_background(g):
    # uneven lighting (morphology notes): an opening removes bright objects the structuring
    # element does not fit in; with a square bigger than the card only the background is left.
    # image - opening = top-hat, the background becomes ~0 and the card stands out evenly
    small = cv2.resize(g, (g.shape[1] // BG_SCALE, g.shape[0] // BG_SCALE), interpolation=cv2.INTER_AREA)
    d = int(BG_SE * min(small.shape)) | 1
    bg = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((d, d), np.uint8))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), d / 4)          # smooth the blocky estimate
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(g.astype(np.float32) - bg, 0, 255).astype(np.uint8)


def midrange(img):
    # halfway between the darkest and the brightest pixel
    # (same as Lyle's linear stretch to 0 - 255 followed by T = 128)
    return float(img.min()) + (float(img.max()) - float(img.min())) / 2


def estimate_angle(g):
    # gradient direction histogram, weighted by magnitude; the card edges give the peak
    s = min(1.0, ANGLE_SIZE / max(g.shape))
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    sm = cv2.GaussianBlur(small.astype(np.float32), (0, 0), ANGLE_SIGMA)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    phi = np.degrees(np.arctan2(gy, gx)) % 90     # the 4 edges (90 deg apart) fall in one bin
    bins = 180
    h, e = np.histogram(phi, bins=bins, range=(0, 90), weights=mag)
    k = int(np.argmax(h))
    centers = (e[:-1] + e[1:]) / 2
    d = np.arange(-3, 4)                           # refine with the 3 bins on each side
    a = (h[(k + d) % bins] * (centers[k] + d * 90 / bins)).sum() / h[(k + d) % bins].sum()
    return a if a < 45 else a - 90


def rotate(img, angle, interp=cv2.INTER_LINEAR):
    # rotate about the center (geometric transforms notes), canvas grown so nothing is cut,
    # new area filled with black so it counts as background
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def find_card(g):
    c = remove_background(g)
    angle = estimate_angle(c)
    r = rotate(c, angle)
    b = r > midrange(r)
    cols, rows = b.sum(0), b.sum(1)               # white pixels per column / row
    x = np.nonzero(cols > 0.5 * cols.max())[0]
    y = np.nonzero(rows > 0.5 * rows.max())[0]
    if len(x) < 10 or len(y) < 10:
        return None, angle, None, None
    box = (y[0], y[-1] + 1, x[0], x[-1] + 1)
    # CHANGED (v5): also a 4 x bigger copy, the corner index is cut from it (keeps the detail of the photo)
    return cut_card(r, box), angle, box, cut_card(r, box, UP)


def cut_card(r, box, scale=1):
    # crop the box from a rotated picture, portrait, normalized size (also used for the colour view)
    y0, y1, x0, x1 = box
    card = np.ascontiguousarray(r[y0:y1, x0:x1])
    if card.shape[1] > card.shape[0]:             # landscape -> portrait
        card = np.ascontiguousarray(np.rot90(card))
    size = (CARD_W * scale, CARD_H * scale)
    enlarge = scale > 1 and size[1] > card.shape[0]          # the big copy of a small card: smooth enlargement
    return cv2.resize(card, size, interpolation=cv2.INTER_CUBIC if enlarge else cv2.INTER_AREA)


# ---------------
# %% Connected components (CCA notes, Lyle's version)
def connected_components(A, B=np.ones((3, 3), np.uint8)):
    # X_k = (X_k-1 dilated by B) intersect A until X_k = X_k-1, then XOR it out of A
    # B = 3 x 3 square -> 8-connected, cross -> 4-connected
    A = A.copy()
    objects = []
    while A.any():
        X = np.zeros_like(A)
        X[tuple(np.argwhere(A)[0])] = 1
        while True:
            Xn = cv2.dilate(X, B) & A
            if (Xn == X).all():
                break
            X = Xn
        objects.append(X)
        A = A ^ X
    return objects


def bbox(X):
    ys, xs = np.nonzero(X)
    return ys.min(), ys.max(), xs.min(), xs.max()


# ---------------
# %% Step 4-5: symbols on the card, face card or not, pip count
def card_symbols(card, boxes):
    # boxes = [(y0, y1, x0, x1), ...] of the two indexes (top-left and bottom-right), they are not pips
    m = cv2.medianBlur(card, 3)
    b = remove_frame_lines((m < midrange(m)).astype(np.uint8), FRAME_K, FRAME_L)   # dark symbols = 1, frame lines removed
    objs = []
    for X in connected_components(b):
        y0, y1, x0, x1 = bbox(X)
        area = int(X.sum())
        if y0 == 0 or x0 == 0 or y1 == CARD_H - 1 or x1 == CARD_W - 1:
            continue                              # card edge / background
        # CHANGED (Lyle): the old fixed corner boxes are replaced by where the index was found
        # in_tl = x1 < CORNER_W * CARD_W and y1 < CORNER_H * CARD_H
        # in_br = x0 > (1 - CORNER_W) * CARD_W and y0 > (1 - CORNER_H) * CARD_H
        # CHANGED (Lyle): the middle of the object has to be inside the box (the bottom-right index is not always an exact mirror)
        if any(by0 <= (y0 + y1) / 2 <= by1 and bx0 <= (x0 + x1) / 2 <= bx1 for by0, by1, bx0, bx1 in boxes):
            continue                              # corner index (read separately)
        objs.append(dict(area=area, box=(y0, y1, x0, x1), mask=X[y0:y1 + 1, x0:x1 + 1]))
    return objs, b


# Gu's original face rule (commented out, replaced by busy_middle below)
# def is_face(objs):
#     if not objs:
#         return False, 0.0, 0.0
#     big = max(objs, key=lambda o: o['area'])
#     y0, y1, x0, x1 = big['box']
#     h, w = (y1 - y0 + 1) / CARD_H, (x1 - x0 + 1) / CARD_W
#     return (h > FACE_H and w > FACE_W), h, w


# CHANGED (Lyle): new face rule (commented out in v5, replaced by picture_frame below: it measures how busy
# the middle is, not whether there is a picture; the Bicycle ace of spades has 18 switches)
# def busy_middle(card):
#     # a number card has only a few pips down its middle, a face picture goes black / white many times
#     m = cv2.medianBlur(card, 3)
#     b = remove_frame_lines((m < midrange(m)).astype(np.uint8), FRAME_K, FRAME_L).astype(np.int8)
#     H, W = b.shape
#     band = b[int(BUSY_Y[0] * H):int(BUSY_Y[1] * H), int(BUSY_X[0] * W):int(BUSY_X[1] * W)]
#     n = float(np.median((np.diff(band, axis=0) != 0).sum(0)))      # switches down each column, the middle one
#     return n > BUSY_MAX, n


# CHANGED (v5): face card = the picture is inside a frame
def picture_frame(card):
    # the picture of a J, Q, K is drawn inside a rectangular frame: a straight vertical line about 80 % of the
    # card height. On a number card or an ace nothing dark runs that far (the longest is the big ace pip, < 45 %).
    m = cv2.medianBlur(card, 3)
    lo, hi = float(m.min()), float(m.max())
    b = (m < lo + FRAME_LEVEL * (hi - lo)).astype(np.uint8)   # a thin line is blurred to gray -> threshold a bit lighter
    b = cv2.dilate(b, np.ones((1, LINE_TOL), np.uint8))      # sideways only: a slightly slanted line stays one run
    H, W = b.shape
    b = b[int(EDGE * H):H - int(EDGE * H), int(EDGE * W):W - int(EDGE * W)]   # not the card edge
    run = best = np.zeros(b.shape[1], int)
    for row in b:                                      # length of the dark run down every column
        run = (run + 1) * row
        best = np.maximum(best, run)
    n = best.max() / H                                 # longest vertical line, fraction of the card height
    return n > FRAME_FACE, n


# CHANGED (v5): a long line can also be a box around the pips of a number card. Then everything inside are pips:
# two or more objects with the shape of the index suit symbol. A face picture never breaks into such pieces.
def filled(m):
    # the object with its holes filled (flood fill the background from outside)
    p = np.pad(m.astype(np.uint8), 1)
    outside = p.copy()
    cv2.floodFill(outside, None, (0, 0), 1)
    return (p | (1 - outside))[1:-1, 1:-1]


def square(m):
    # the filled shape in the middle of a square (keeps its height / width), scaled to SHAPE_N x SHAPE_N
    m = filled(m)
    h, w = m.shape
    n = max(h, w)
    sq = np.zeros((n, n), np.uint8)
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = m
    return cv2.resize(sq.astype(np.float32), (SHAPE_N, SHAPE_N), interpolation=cv2.INTER_AREA) > 0.5


def same_shape(a, b):
    # overlap (IoU) of two shapes scaled to the same size, the better of the two ways up
    a, b = square(a), square(b)
    iou = lambda u, v: (u & v).sum() / max(1, (u | v).sum())
    return max(iou(a, b), iou(a[::-1, ::-1], b))


def boxed_pips(found, suit):
    sims = [same_shape(o['mask'], suit) for o in found]
    return len(found) >= 2 and min(sims) >= PIP_SHAPE, (min(sims) if sims else 0.0)


def pips(objs):
    if not objs:
        return []
    biggest = max(o['area'] for o in objs)
    found = []
    for o in objs:
        y0, y1, x0, x1 = o['box']
        h, w = y1 - y0 + 1, x1 - x0 + 1
        if o['area'] >= max(PIP_REL * biggest, PIP_MIN * CARD_W * CARD_H) and max(h, w) <= PIP_LONG * min(h, w):
            found.append(o)                       # CHANGED (v5): a pip is not a line (leftover of a frame or box)
    return found


# ---------------
# %% Step 6: corner index -> rank symbol and suit symbol
def remove_frame_lines(b, k, L):
    # the thin straight frame of the face pictures can touch the index symbols;
    # top-hat (A minus its opening) keeps only thin parts, opening with a long line keeps only
    # the long straight ones -> remove those
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    thin = b & (1 - cv2.morphologyEx(b, cv2.MORPH_OPEN, disk))
    lines = (cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8)) |
             cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)))
    return b & (1 - cv2.dilate(lines, np.ones((3, 3), np.uint8)))


# Gu's original: split_corner cut a fixed box (top-left 21 % x 29 %) and took the first two objects in it.
# CHANGED (Lyle): find_index + split_index below find the index as the leftmost objects on the card,
# so an index bigger or smaller than the old box still works.
def find_index(card):
    # the index = the leftmost object (rank or suit) plus the objects stacked with it (rank above suit,
    # "10" = two objects next to each other); the frame lines of the face pictures are removed first
    m = cv2.medianBlur(card, 3)
    b = remove_frame_lines((m < midrange(m)).astype(np.uint8), FRAME_K, FRAME_L)
    h, w = int(INDEX_ROWS * CARD_H), int(INDEX_COLS * CARD_W)
    b = b[:h, :w]                                                # only the top-left part is searched (faster)
    small = MIN_AREA * CORNER_W * CORNER_H * CARD_W * CARD_H
    objs = []
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))   # 4-connected: a frame line touching a symbol only diagonally stays separate
    for X in connected_components(b, cross):
        y0, y1, x0, x1 = bbox(X)
        if X.sum() < small or y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1:
            continue                                             # noise, card edge, big things cut by the part
        objs.append([y0, y1, x0, x1, X])
    if not objs:
        return None
    objs.sort(key=lambda o: o[2])                                # leftmost first
    group = [objs.pop(0)]
    gap = max(2, int(round(INDEX_GAP * CARD_W)))
    grew = True
    while grew:
        grew = False
        tall = max(o[1] - o[0] + 1 for o in group)
        for o in objs:
            for g in group:
                stacked = o[2] <= g[3] and o[3] >= g[2] and o[0] <= g[1] + tall and o[1] >= g[0] - tall
                # same row of symbols ("10"): same top and bottom, just a small gap in between
                beside = abs(o[0] - g[0]) <= gap and abs(o[1] - g[1]) <= gap and o[2] <= g[3] + gap and o[3] >= g[2] - gap
                if stacked or beside:
                    group.append(o)
                    objs.remove(o)
                    grew = True
                    break
            if grew:
                break
    mask = np.zeros((CARD_H, CARD_W), np.uint8)
    for o in group:
        mask[:h, :w] |= o[4]
    return mask


def split_corner(card):
    # Gu's original, now only the last resort when the leftmost objects can not be split into rank and suit:
    # fixed box (top-left 21 % x 29 %, enlarged 4 x), frame lines removed, first two objects from the top
    roi = card[:int(CORNER_H * CARD_H), :int(CORNER_W * CARD_W)]
    roi = cv2.resize(roi, (roi.shape[1] * UP, roi.shape[0] * UP), interpolation=cv2.INTER_CUBIC)
    m = cv2.medianBlur(roi, 3)
    H, W = m.shape
    b = remove_frame_lines((m < midrange(m)).astype(np.uint8), int(round(FRAME_DISK * W)) | 1, int(round(FRAME_LINE * H)))
    found = []
    for X in connected_components(b):
        y0, y1, x0, x1 = bbox(X)
        if X.sum() < MIN_AREA * H * W:                           # noise
            continue
        if y0 == 0 or x0 == 0 or y1 == H - 1 or x1 == W - 1:
            continue                                             # background, card edge, pips cut by the box
        found.append([y0, y1, X])
    found.sort(key=lambda o: o[0])
    rows = []
    for y0, y1, X in found:
        if rows and y0 <= rows[-1][1]:
            rows[-1][1] = max(rows[-1][1], y1)
            rows[-1][2] = rows[-1][2] | X
        else:
            rows.append([y0, y1, X])
    symbols = []
    for _, _, X in rows[:2]:
        ys0, ys1, xs0, xs1 = bbox(X)
        symbols.append(X[ys0:ys1 + 1, xs0:xs1 + 1])
    return symbols, b


def index_box(mask):
    # box around the index objects (with the margin), (y0, y1, x0, x1)
    y0, y1, x0, x1 = bbox(mask)
    return y0 - INDEX_PAD, y1 + INDEX_PAD, x0 - INDEX_PAD, x1 + INDEX_PAD


def mirror_box(box):
    # the same box seen from the opposite corner (card turned 180 deg)
    y0, y1, x0, x1 = box
    return CARD_H - 1 - y1, CARD_H - 1 - y0, CARD_W - 1 - x1, CARD_W - 1 - x0


def split_index(card, mask, big=None):
    ys0, ys1, xs0, xs1 = bbox(mask)
    y0, y1 = max(0, ys0 - INDEX_PAD), min(CARD_H, ys1 + 1 + INDEX_PAD)
    x0, x1 = max(0, xs0 - INDEX_PAD), min(CARD_W, xs1 + 1 + INDEX_PAD)
    size = ((x1 - x0) * UP, (y1 - y0) * UP)
    if big is None:
        roi = cv2.resize(card[y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC)
    else:                                                        # CHANGED (v5): cut from the 4 x copy (more detail)
        roi = big[y0 * UP:y1 * UP, x0 * UP:x1 * UP]
    keep = cv2.dilate(cv2.resize(mask[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST),
                      np.ones((UP + 1, UP + 1), np.uint8))       # only the index objects, nothing touching them
    m = cv2.medianBlur(roi, 3)
    b = (m < midrange(m)).astype(np.uint8) & keep                # symbols = 1
    small = MIN_AREA * CORNER_W * CORNER_H * CARD_W * CARD_H * UP * UP
    found = []
    for X in connected_components(b):
        if X.sum() < small:                                      # noise
            continue
        yy0, yy1, _, _ = bbox(X)
        found.append([yy0, yy1, X])
    found.sort(key=lambda o: o[0])
    rows = []                                                    # objects sharing rows = one symbol ("10")
    for yy0, yy1, X in found:
        if rows and yy0 <= rows[-1][1]:
            rows[-1][1] = max(rows[-1][1], yy1)
            rows[-1][2] = rows[-1][2] | X
        else:
            rows.append([yy0, yy1, X])
    symbols = []
    for _, _, X in rows[:2]:
        a0, a1, c0, c1 = bbox(X)
        symbols.append(X[a0:a1 + 1, c0:c1 + 1])
    return symbols, b


# ---------------
# %% Step 7: suit from the row-width profile
def suit_features(m):
    w = np.convolve(m.sum(1).astype(float), np.ones(5) / 5, 'same')     # width of every row
    top = w[:max(1, round(0.15 * len(w)))].mean() / w.max()             # how wide the top is
    asym = np.abs(w - w[::-1]).mean() / w.max()                         # 0 = top/bottom mirror
    return top, asym


def suit_rule(top, asym):
    if asym < ASYM_DIAMOND:          # the diamond is the only suit that looks the same upside down
        return 'd', top, asym
    if top > TOP_HEART:
        return 'h', top, asym
    if top > TOP_CLUB:
        return 'c', top, asym
    return 's', top, asym


def classify_suit(m):
    return suit_rule(*suit_features(m))


# CHANGED (v5): on a number card the suit is read from the pips, they are 2 - 3 x bigger than the index symbol
def pip_suit(found):
    # pips in the lower half are printed upside down -> turned back; median of the features of all pips
    feats = []
    for o in found:
        y0, y1, _, _ = o['box']
        m = o['mask'] if (y0 + y1) / 2 <= PIP_UPRIGHT * CARD_H else o['mask'][::-1, ::-1]
        feats.append(suit_features(m))
    return suit_rule(float(np.median([f[0] for f in feats])), float(np.median([f[1] for f in feats])))


# ---------------
# %% Step 8: J / Q / K from the structure of the index letter
def count_holes(m):
    # holes = background objects that do not reach the border (4-connected background)
    p = np.pad(1 - m, 1, constant_values=1).astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    n = 0
    for X in connected_components(p, cross):
        y0, y1, x0, x1 = bbox(X)
        if y0 > 0 and x0 > 0 and y1 < p.shape[0] - 1 and x1 < p.shape[1] - 1 and X.sum() >= HOLE_MIN * m.size:
            n += 1
    return n


def face_letter(m):
    holes = count_holes(m)
    col = m.sum(0)
    stem = np.argmax(col) / max(1, m.shape[1] - 1)    # where the long vertical stroke is (0 = left)
    if holes >= 1:
        return 'Q', holes, stem
    return ('K' if stem < STEM_K else 'J'), holes, stem


# ---------------
# %% Whole card
def read_index(card, big=None):
    # top-left index; if it can not be split, turn the card 180 deg and use the bottom-right one
    # returns the two symbols, the picture of the index, which corner, and the boxes of both indexes
    # big = the 4 x copy of the card (CHANGED (v5)), the index is cut from it
    turn = lambda a: None if a is None else np.ascontiguousarray(np.rot90(a, 2))
    turned = turn(card)
    corners = [('top-left', card, big, find_index(card)), ('bottom-right', turned, turn(big), find_index(turned))]
    boxes = [index_box(mk) if i == 0 else mirror_box(index_box(mk))
             for i, (_, _, _, mk) in enumerate(corners) if mk is not None]
    if not boxes:
        boxes = [(0, int(CORNER_H * CARD_H), 0, int(CORNER_W * CARD_W))]          # nothing found: the old fixed corners
    boxes += [mirror_box(b) for b in boxes]                                       # the other corner (the found boxes can be off)
    first = None
    for name, c, bc, mask in corners:
        if mask is None:
            continue
        symbols, b = split_index(c, mask, bc)
        if len(symbols) == 2:
            return symbols, b, name, boxes
        first = first or (symbols, b, 'none', boxes)
    for name, c, _, _ in corners:                                # last resort: Gu's fixed box
        symbols, b = split_corner(c)
        if len(symbols) == 2:
            return symbols, b, name + ' (fixed box)', boxes
    return first or ([], np.zeros((1, 1), np.uint8), 'none', boxes)


def recognize(img):
    res = dict(ok=False)
    g = to_gray(img)
    card, angle, box, big = find_card(g)
    res['angle'] = angle
    if card is None:
        return res
    res['card'], res['box'] = card, box
    symbols, res['index_binary'], res['corner'], boxes = read_index(card, big)
    objs, res['symbols_binary'] = card_symbols(card, boxes)
    # Gu's original: res['face'], res['face_h'], res['face_w'] = is_face(objs)
    # res['face'], res['busy_n'] = busy_middle(card)       # CHANGED (Lyle): face card from the middle of the card
    res['face'], res['frame'] = picture_frame(card)        # CHANGED (v5): face card = picture frame
    if len(symbols) == 2:
        res['rank_img'], res['suit_img'] = symbols
        res['suit'], res['top'], res['asym'] = classify_suit(symbols[1])
        res['suit_from'] = 'index'
        if res['face']:            # CHANGED (v5): a frame with only pips inside = a box around the pips
            boxed, res['pip_shape'] = boxed_pips(pips(objs), symbols[1])
            res['face'] = not boxed
    if res['face']:
        if 'rank_img' in res:
            res['rank'], res['holes'], res['stem'] = face_letter(res['rank_img'])
    else:
        res['pips'] = pips(objs)
        n = len(res['pips'])
        if 1 <= n <= 10:
            res['rank'] = 'A' if n == 1 else str(n)
        if n >= 2:      # CHANGED (v5): suit from the pips (an ace's single middle pip can be either way up -> index)
            res['suit'], res['top'], res['asym'] = pip_suit(res['pips'])
            res['suit_from'] = 'pips'
    res['ok'] = 'rank' in res and 'suit' in res
    return res


def describe(res):
    rank = RANK_NAMES.get(res.get('rank'), res.get('rank', '?'))
    suit = SUIT_NAMES.get(res.get('suit'), '?')
    if not res.get('ok'):
        return 'Could not read the card (rank %s, suit %s)' % (rank, suit)
    return '%s of %s' % (rank, suit)


# ---------------
# %% Output: window with the straightened card, and the result printed
def show(img, res):
    title = describe(res)
    print('\nResult: %s' % title)
    if 'card' not in res:
        print('  card not found')
        return
    how = ('face card (frame line %.2f of the card height), index letter: %d hole(s), long stroke at %.2f of the width'
           % (res['frame'], res.get('holes', -1), res.get('stem', -1)) if res['face']
           else 'number card (longest line %.2f of the card height%s), %d pip(s) counted'
           % (res['frame'], ', a box: only pips inside' if res['frame'] > FRAME_FACE else '', len(res.get('pips', []))))
    src = 'pips' if res.get('suit_from') == 'pips' else 'the %s index' % res['corner']
    print('  tilt %.1f deg, %s;  suit from %s (top width %.2f, asymmetry %.2f)'
          % (res['angle'], how, src, res.get('top', -1), res.get('asym', -1)))

    # straightened card in colour (only for showing it), with what was counted
    view = cut_card(rotate(img, res['angle']), res['box']) if img.ndim == 3 else res['card']
    view = cv2.resize(view, (2 * CARD_W, 2 * CARD_H), interpolation=cv2.INTER_CUBIC)
    plt.figure(figsize=(12, 6.5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap='gray')
    plt.title('Input'); plt.axis('off')
    ax = plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB) if view.ndim == 3 else view, cmap='gray')
    for o in res.get('pips', []):
        y0, y1, x0, x1 = [2 * v for v in o['box']]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='lime', lw=2))
    plt.title('Straightened card (tilt %.1f deg)\n%s' % (res['angle'], 'face card' if res['face']
              else '%d pips' % len(res.get('pips', [])))); plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(res['index_binary'], cmap='gray')
    plt.title('%s index' % res['corner']); plt.axis('off')
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()


def main():
    print('1. Acquire from Webcam')
    print('2. Read from File')
    choice = ask('Choose option:  ')
    if choice == '1':
        cap = open_camera()
        while True:                                # SPACE = picture, q = quit
            f = capture(cap)
            if f is None:
                break
            show(f, recognize(f))                  # close the window to take the next picture
        cap.release()
        cv2.destroyAllWindows()
    elif choice == '2':
        while True:                                # one image after another, q = quit
            f = read_file()
            if f is None:
                break
            show(f, recognize(f))                  # close the window to read the next image
    else:
        raise SystemExit('Invalid option')
    print('Bye')


if __name__ == '__main__':
    main()
