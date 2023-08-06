import cv2


def plot(frame, lbs, alpha=.5):
    pt1 = [10, 10]
    start_hint_box = pt1[1]

    fonScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    frameC = frame.copy()
    lbs = lbs.split("\n")
    lbs = [x for x in lbs if len(x) > 1]
    for idx, string in enumerate(lbs):
        fontWidth, fontHeight = cv2.getTextSize(string, fontFace=font, fontScale=fonScale, thickness=1)[0]

        pt2 = [pt1[0] + fontWidth, int(pt1[1] * 2 + fontHeight)]
        pt2[1] = pt2[1] * (idx + 1)

        frame = cv2.rectangle(frame, [pt1[0], start_hint_box], pt2, (0, 0, 0), thickness=cv2.FILLED)
        frame = cv2.putText(frame, string, org=(pt1[0], pt2[1] - pt1[1]), fontFace=font,
                            color=(255, 255, 255), fontScale=fonScale, thickness=1)
        start_hint_box = pt2[1]
    frame = cv2.addWeighted(frame, alpha, frameC, 1 - alpha, gamma=0)

    return frame
