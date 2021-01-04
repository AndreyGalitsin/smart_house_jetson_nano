import cv2
import face_recognition
import numpy as np
import sys

class FaceRec:
    def __init__(self):
        self.img_path_arr = [
        "face_1.jpg", 
        #"face_2.jpg"
        ]

        self.name_arr = [
            "Andrey",
            #"Sergey"
            ]

        self.known_faces, self.known_face_names = self.get_faces_database()

    def get_faces_database(self):  
        known_faces = []
        known_face_names = []

        for i in range(len(self.img_path_arr)):
        #for i in range(1):
            image = face_recognition.load_image_file(self.img_path_arr[i])
            face_encoding = face_recognition.face_encodings(image)[0]

            known_faces.append(face_encoding)
            known_face_names.append(self.name_arr[i])

        return known_faces, known_face_names

    def label_results(self, frame, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):          
            if not name:            
                continue    
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)         
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
                
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        return frame

    def get_face_locations(self, rgb_frame):
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

        return face_locations

    def get_face_names(self,rgb_frame, face_locations):
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        return face_names

    def get_face_rec(self, frame):
        rgb_frame = frame[:, :, ::-1]

        face_locations = self.get_face_locations(rgb_frame)
        face_names = self.get_face_names(rgb_frame, face_locations)
        last_frame = self.label_results(frame, face_locations, face_names)
        return last_frame, face_names

    def show_result(self, last_frame):
        cv2.imshow('Face recognition', last_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 	
            sys.exit()

    def identificate_person(self, face_names):
        if len(face_names) > 0:
            if not 'Unknown' in face_names:
                #turn signalisation off
                print('turn signalisation off')
                return 1
            else:
                return 0

    def main(self):
        cam = cv2.VideoCapture(0)
        process_this_frame = True
        
        while True:
            _, frame = cam.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            frame = small_frame

            if process_this_frame:
                last_frame, face_names = self.get_face_rec(frame)
            
            self.show_result(last_frame)
            identification = self.identificate_person(face_names)
            process_this_frame = not process_this_frame

        cv2.destroyAllWindows()

    def main_for_img(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame = small_frame

        last_frame, face_names = self.get_face_rec(frame)
        identification = self.identificate_person(face_names)

        return last_frame, identification

if __name__ == "__main__":
    face_rec = FaceRec()
    #face_rec.main()

    cam = cv2.VideoCapture(0)
    process_this_frame = True
    while True:
        _, frame = cam.read()
        if process_this_frame:
            last_frame, identification = face_rec.main_for_img(frame)
        cv2.imshow('Face recognition', last_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 	
            break
        process_this_frame = not process_this_frame
    cv2.destroyAllWindows()

    




