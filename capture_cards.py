# side tool, not part of the project script: take card pictures with the same camera setup as Lyle_Gu_P4.py
# SPACE = take picture (asks for a name, blank = timestamp), + / - = focus, q = quit
# the autofocus is off so the focus is stuck where it was, use + / - until the card is sharp, the value is printed when you quit
# saved as card_<name>.png next to this script, so they show up in option 2 (Read from File)
import os
import time
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # work from the script's folder

# ---------------
# %% Settings (keep the same as Lyle_Gu_P4.py)
ZOOM_PCT = 70        # 1.7 zoom
CAM_W, CAM_H = 1280, 720   # webcam resolution, default is 640 x 480
AVG_FRAMES = 10      # number of frames averaged for the picture
focus = 70            # manual focus, 0 - 255, changed live with + / -

def zoom(img):
    # Just artifically zooming in the webcam on the setup, not actual detection processing yet
    # basically crops to center
    h, w = img.shape[:2]
    s = 1 + ZOOM_PCT / 100
    ch, cw = round(h / s), round(w / s)
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC)

# ---------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit('Could not open camera')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)    # more pixels on the card
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # fixed focus, set after the resolution since changing it can reset the focus
focusOk = cap.set(cv2.CAP_PROP_FOCUS, focus)
msg = ''
while True:
    ok, f = cap.read()
    f = zoom(f)
    cv2.putText(f, 'SPACE = take picture   q = quit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(f, msg, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(f, ('focus %d   + / - to change' % focus) if focusOk else 'camera has no manual focus', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Preview', f)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        for _ in range(15):                     # let auto exposure settle
            ok, f = cap.read()
            cv2.waitKey(1)                      # keeps the window from saying "not responding"

        # Frame averaging (restoration notes): g_bar = (1/M) * sum of g_i
        # the card and camera stay still so only the noise changes and it cancels out
        total = np.zeros(f.shape)
        for _ in range(AVG_FRAMES):
            ok, f = cap.read()
            cv2.waitKey(1)
            total += f
        f = zoom(np.round(total / AVG_FRAMES).astype(np.uint8))

        name = input('Name for this picture (like 6D, blank = timestamp):  ').strip()
        fileName = 'card_%s.png' % (name or time.strftime('%Y%m%d_%H%M%S'))
        cv2.imwrite(fileName, f)
        msg = 'saved ' + fileName
        print(msg)
    if key in (ord('+'), ord('=')):
        focus = min(focus + 5, 255)
        focusOk = cap.set(cv2.CAP_PROP_FOCUS, focus)
    if key == ord('-'):
        focus = max(focus - 5, 0)
        focusOk = cap.set(cv2.CAP_PROP_FOCUS, focus)
    if key == ord('q'):
        break
print('last focus value: %d' % focus)
cap.release()
cv2.destroyAllWindows()
