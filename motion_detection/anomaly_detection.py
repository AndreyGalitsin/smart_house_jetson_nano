import cv2 # импорт модуля cv2
import datetime

class MoveRec:
    def __init__(self):
        self.th = 50
        self.area = 50

    def main(self, frame1, frame2):
        diff = cv2.absdiff(frame1, frame2) # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # перевод кадров в черно-белую градацию
        blur = cv2.GaussianBlur(gray, (5, 5), 0) # фильтрация лишних контуров
        _, thresh = cv2.threshold(blur, self.th, 255, cv2.THRESH_BINARY) # метод для выделения кромки объекта белым цветом
        dilated = cv2.dilate(thresh, None, iterations = 3) # данный метод противоположен методу erosion(), т.е. эрозии объекта, и расширяет выделенную на предыдущем этапе область
        _, сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # нахождение массива контурных точек
        if len(сontours) != 0:
            status = 'Motion detected'
        else:
            status = None
        for contour in сontours:
            (x, y, w, h) = cv2.boundingRect(contour) # преобразование массива из предыдущего этапа в кортеж из четырех координат
            # метод contourArea() по заданным contour точкам, здесь кортежу, вычисляет площадь зафиксированного объекта в каждый момент времени, это можно проверить
            if cv2.contourArea(contour) < self.area: # условие при котором площадь выделенного объекта меньше 700 px
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2) # получение прямоугольника из точек кортежа
            cv2.putText(frame1, "Status: {}".format("Motion detected"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # вставляем текст
        return frame1, status


if __name__ == "__main__":
    move_rec = MoveRec()

    cam = cv2.VideoCapture(0)
    _, frame1 = cam.read()
    _, frame2 = cam.read()
    counter = 0
    while True:
        counter += 1

        if counter % 50 == 0:
            counter = 0
            if frame1 is not None:
                frame1, status = move_rec.main(frame1, frame2)
                #print(status)

                cv2.imshow('Motion recognition', frame1)
                frame1 = frame2  #
                _, frame2 = cam.read()

                if cv2.waitKey(1) & 0xFF == ord('q'): 	
                    break
            else:
                print("cannot receive img from camera")
                cam = cv2.VideoCapture(0)
                time.sleep(0.01)
        else:
            print('qqq')
            continue
    cv2.destroyAllWindows()

