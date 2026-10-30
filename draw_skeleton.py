# side tool, not part of the project script: draw an upright skeleton with the mouse and get the points to hard code
# hold left mouse button to draw, c = clear, s = save (asks for a name), q = quit
# prints skel_<name> = np.array([...]) and adds it to skeletons.txt, paste that into Lyle_Gu_P4.py
# the symbols are stretched to fill the 64 x 64 box, so reach the leaves toward the red box and the stem toward the bottom
import os
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # save next to the script

SIZE = 64            # same as the symbols in Lyle_Gu_P4.py
BIG = 512            # drawing size, shrunk to SIZE after
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
        name = input('Name for this skeleton (club, spade, ...):  ').strip()
        pts = np.argwhere(cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 60)
        text = 'skel_%s = np.array([%s])   # (row, col), %d points\n' % (
            name, ', '.join('(%d, %d)' % (r, c) for r, c in pts), len(pts))
        print(text)
        with open('skeletons.txt', 'a') as fh:
            fh.write(text)
        canvas[:] = 0                                      # ready for the next one
    if key == ord('q'):
        break
cv2.destroyAllWindows()
