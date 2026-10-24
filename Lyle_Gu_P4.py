# clear; close all; clc;
import os
import glob
import time
import cv2

# %% Settings
ZOOM_PCT = 70        # digital zoom, 20 = 1.2x center crop, 0 = off
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
        ok, f = cap.read()
        if not ok:
            raise SystemExit('Camera gave no frame')
        f = zoom(f)
        cv2.putText(f, 'SPACE = take picture   q = quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Preview', f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            for _ in range(10):      # let auto exposure settle
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

# %% Output
cv2.imshow('Input Image', f)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% Image processing goes here
