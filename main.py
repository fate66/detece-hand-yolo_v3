from detece import predict, initModel
# import sys
import cv2

# print(sys.argv)

# base = initModel()
# res = predict(sys.argv[1], base)
# print(res)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    base = initModel('left')
    while (True):
        ret, frame = cap.read()
        # print(frame)
        res = predict(frame, base)
        print(res)
        if not ret:
            break
        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按键盘q就停止拍照
            break
    cap.release()
    cv2.destroyAllWindows()
