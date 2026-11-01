"""
ECE 4367 / 5367  Image Processing  -  Project 4   (version 2: no templates)
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
       Crop, turn to portrait, resize to 250 x 350.
    4. Symbols on the card: threshold halfway between darkest and brightest, connected component
       analysis (dilate-intersect-XOR from the notes); drop objects touching the card border and
       the corner index objects.
    5. Face card or number card: a face picture is one big object spanning more than 60 % of the
       card height and 50 % of its width; pips are much smaller (even the big ace of spades).
    6. Corner index (top-left 21 % x 29 %, enlarged 4 x): threshold, remove the long thin frame
       lines of the face pictures (white top-hat + opening with long line structuring elements),
       connected components, drop objects touching the border or tiny, merge objects that share
       rows ("10"); top object = rank symbol, next = suit symbol.  If this fails the card is
       turned 180 deg and the bottom-right index is used.
    7. Suit from the row-width profile of the suit symbol:
         top 15 % of the rows wide (two lobes)      -> heart
         top 15 % medium (one round lobe)            -> club
         top/bottom mirror symmetric                 -> diamond,  otherwise spade
    8. Rank
         number card: count the pips (objects at least 40 % of the biggest one): 1 = A, 2 - 10
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
    - The thresholds (face picture size, J/K stroke position, suit rules) were set on our deck.
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
CORNER_W, CORNER_H = 0.21, 0.29  # corner index region, fraction of the card
UP = 4                           # corner enlargement
MIN_AREA = 0.003                 # corner objects smaller than this fraction of the corner are noise
FRAME_DISK = 0.10                # top-hat disk, fraction of the corner width
FRAME_LINE = 0.40                # line length, fraction of the corner height

FACE_H, FACE_W = 0.60, 0.50      # face picture: one object taller / wider than this part of the card
PIP_REL = 0.40                   # pip: at least 40 % of the biggest object on the card ...
PIP_MIN = 0.003                  # ... and at least 0.3 % of the card
HOLE_MIN = 0.02                  # a hole in a letter is at least 2 % of the letter's box
STEM_K = 0.40                    # fullest column left of 40 % of the letter width -> K, else J

TOP_HEART = 0.47                 # suit rules, set from the 52 photos in cards/named
TOP_CLUB = 0.235
ASYM_DIAMOND = 0.115

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
        return None, angle, None
    box = (y[0], y[-1] + 1, x[0], x[-1] + 1)
    return cut_card(r, box), angle, box


def cut_card(r, box):
    # crop the box from a rotated picture, portrait, normalized size (also used for the colour view)
    y0, y1, x0, x1 = box
    card = r[y0:y1, x0:x1]
    if card.shape[1] > card.shape[0]:             # landscape -> portrait
        card = np.rot90(card)
    return cv2.resize(np.ascontiguousarray(card), (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)


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
def card_symbols(card):
    m = cv2.medianBlur(card, 3)
    b = (m < midrange(m)).astype(np.uint8)       # dark symbols = 1
    objs = []
    for X in connected_components(b):
        y0, y1, x0, x1 = bbox(X)
        area = int(X.sum())
        if y0 == 0 or x0 == 0 or y1 == CARD_H - 1 or x1 == CARD_W - 1:
            continue                              # card edge / background
        in_tl = x1 < CORNER_W * CARD_W and y1 < CORNER_H * CARD_H
        in_br = x0 > (1 - CORNER_W) * CARD_W and y0 > (1 - CORNER_H) * CARD_H
        if in_tl or in_br:
            continue                              # corner index (read separately)
        objs.append(dict(area=area, box=(y0, y1, x0, x1)))
    return objs, b


def is_face(objs):
    if not objs:
        return False, 0.0, 0.0
    big = max(objs, key=lambda o: o['area'])
    y0, y1, x0, x1 = big['box']
    h, w = (y1 - y0 + 1) / CARD_H, (x1 - x0 + 1) / CARD_W
    return (h > FACE_H and w > FACE_W), h, w


def pips(objs):
    if not objs:
        return []
    biggest = max(o['area'] for o in objs)
    return [o for o in objs if o['area'] >= max(PIP_REL * biggest, PIP_MIN * CARD_W * CARD_H)]


# ---------------
# %% Step 6: corner index -> rank symbol and suit symbol
def remove_frame_lines(b):
    # the thin straight frame of the face pictures can touch the index symbols;
    # top-hat (A minus its opening) keeps only thin parts, opening with a long line keeps only
    # the long straight ones -> remove those
    H, W = b.shape
    k = int(round(FRAME_DISK * W)) | 1
    L = int(round(FRAME_LINE * H))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    thin = b & (1 - cv2.morphologyEx(b, cv2.MORPH_OPEN, disk))
    lines = (cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8)) |
             cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)))
    return b & (1 - cv2.dilate(lines, np.ones((3, 3), np.uint8)))


def split_corner(card):
    roi = card[:int(CORNER_H * CARD_H), :int(CORNER_W * CARD_W)]
    roi = cv2.resize(roi, (roi.shape[1] * UP, roi.shape[0] * UP), interpolation=cv2.INTER_CUBIC)
    m = cv2.medianBlur(roi, 3)
    b = remove_frame_lines((m < midrange(m)).astype(np.uint8))   # symbols = 1
    H, W = b.shape
    found = []
    for X in connected_components(b):
        y0, y1, x0, x1 = bbox(X)
        if X.sum() < MIN_AREA * H * W:                           # noise
            continue
        if y0 == 0 or x0 == 0 or y1 == H - 1 or x1 == W - 1:
            continue                                             # background, card edge, pips cut by the box
        found.append([y0, y1, X])
    found.sort(key=lambda o: o[0])
    rows = []                                                    # objects sharing rows = one symbol ("10")
    for y0, y1, X in found:
        if rows and y0 <= rows[-1][1]:
            rows[-1][1] = max(rows[-1][1], y1)
            rows[-1][2] = rows[-1][2] | X
        else:
            rows.append([y0, y1, X])
    symbols = []
    for y0, y1, X in rows[:2]:
        ys0, ys1, xs0, xs1 = bbox(X)
        symbols.append(X[ys0:ys1 + 1, xs0:xs1 + 1])
    return symbols, b


# ---------------
# %% Step 7: suit from the row-width profile
def suit_features(m):
    w = np.convolve(m.sum(1).astype(float), np.ones(5) / 5, 'same')     # width of every row
    top = w[:max(1, round(0.15 * len(w)))].mean() / w.max()             # how wide the top is
    asym = np.abs(w - w[::-1]).mean() / w.max()                         # 0 = top/bottom mirror
    return top, asym


def classify_suit(m):
    top, asym = suit_features(m)
    if top > TOP_HEART:
        return 'h', top, asym
    if top > TOP_CLUB:
        return 'c', top, asym
    return ('d' if asym < ASYM_DIAMOND else 's'), top, asym


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
def read_index(card):
    # top-left index; if it can not be split, turn the card 180 deg and use the bottom-right one
    symbols, b = split_corner(card)
    if len(symbols) == 2:
        return symbols, b, 'top-left'
    symbols2, b2 = split_corner(np.ascontiguousarray(np.rot90(card, 2)))
    if len(symbols2) == 2:
        return symbols2, b2, 'bottom-right'
    return symbols, b, 'none'


def recognize(img):
    res = dict(ok=False)
    g = to_gray(img)
    card, angle, box = find_card(g)
    res['angle'] = angle
    if card is None:
        return res
    res['card'], res['box'] = card, box
    objs, res['symbols_binary'] = card_symbols(card)
    res['face'], res['face_h'], res['face_w'] = is_face(objs)
    symbols, res['index_binary'], res['corner'] = read_index(card)
    if len(symbols) == 2:
        res['rank_img'], res['suit_img'] = symbols
        res['suit'], res['top'], res['asym'] = classify_suit(symbols[1])
    if res['face']:
        if 'rank_img' in res:
            res['rank'], res['holes'], res['stem'] = face_letter(res['rank_img'])
    else:
        res['pips'] = pips(objs)
        n = len(res['pips'])
        if 1 <= n <= 10:
            res['rank'] = 'A' if n == 1 else str(n)
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
    how = ('face card, index letter: %d hole(s), long stroke at %.2f of the width'
           % (res.get('holes', -1), res.get('stem', -1)) if res['face']
           else '%d pips counted' % len(res.get('pips', [])))
    print('  tilt %.1f deg, %s;  suit from the %s index (top width %.2f, asymmetry %.2f)'
          % (res['angle'], how, res['corner'], res.get('top', -1), res.get('asym', -1)))

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
