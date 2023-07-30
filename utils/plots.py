import cv2


def plot(frame, lbs, alpha=.5):
    pt1 = (10, 10)
    fonScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    frameC = frame.copy()
    fontWidth, fontHeight = cv2.getTextSize(lbs, fontFace=font, fontScale=fonScale, thickness=1)[0]
    pt2 = (pt1[0] + fontWidth, int((pt1[1] + fontHeight) * 1.5))
    frame = cv2.rectangle(frame, pt1, pt2, (0, 0, 0), thickness=cv2.FILLED)
    frame = cv2.addWeighted(frame, alpha, frameC, 1 - alpha, gamma=0)

    frame = cv2.putText(frame, lbs, org=(pt1[0], int((pt1[1] + fontHeight) * 1.2)), fontFace=font,
                        color=(255, 255, 255), fontScale=fonScale, thickness=1)
    return frame
