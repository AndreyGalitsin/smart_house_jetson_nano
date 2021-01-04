import cv2
import face_recognition
import numpy as np

def get_faces_database(img_path_arr, name_arr):  
    known_faces = []
    known_face_names = []

    #for i in range(len(img_path_arr)):
    for i in range(1):
        image = face_recognition.load_image_file(img_path_arr[i])
        print('@@@@@', face_recognition.face_encodings(image))
        face_encoding = face_recognition.face_encodings(image)[0]

        known_faces.append(face_encoding)
        known_face_names.append(name_arr[i])

    return known_faces, known_face_names


def main(img_path_arr, name_arr):
    video_capture = cv2.VideoCapture(0)
    known_faces, known_face_names = get_faces_database(img_path_arr, name_arr)
    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []

    process_this_frame = True
    
    while True:
            
        # Grab a single frame of video
            
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame = small_frame
            
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)    
        rgb_frame = frame[:, :, ::-1]

        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame  
            
        # Label the results      
        for (top, right, bottom, left), name in zip(face_locations, face_names):          
            if not name:            
                continue    
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)         
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
                
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 	
            break

    input_movie.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path_arr = [
    "face_1.jpg", 
    "face_2.jpg"]

    name_arr = [
        "Andrey",
        "Sergey"
        ]

    main(img_path_arr, name_arr)




