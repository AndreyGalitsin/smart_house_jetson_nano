import cv2
import face_recognition
video_capture = cv2.VideoCapture(0)
while 1:
    
    ret, image = video_capture.read()
    
    if len(face_recognition.face_encodings(image)) > 0:
        cv2.imwrite('./face_2.jpg', image)
        print('Done!')
        break
    
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'): 	
        break

video_capture.release()
cv2.destroyAllWindows()
