from ultralytics import YOLO
import os
import cv2


def detect_person(path="/home/tuantv/Desktop/hieu/classroom_selfcollect_dataset/WIN_20230807_14_48_17_Pro.mp4", model_name="yolov8x", conf=0.01, iou=0.1, save_result=True, save_txt=True):
    # load a model
    # model = YOLO("yolov8n.pt")
    
    # load yolov8 model for small object detection
    model = YOLO(model_name+ ".pt")
    
    # check if path exists
    if os.path.exists(path):
        if os.path.isdir(path):
            db = path
            # paths = [os.path.join(db, p) for p in os.listdir(db)]
        elif ".mp4" in path:
            db = read_frames_from_video(path)
        else: #if ".jpg" in path or ".JPG" in path or ".png" in path or ".PNG" in path:
            db = path
    

    # train the model
    # model.train(data="coco128.yaml", epochs=3) 
    # metrics = model.val()

    # detect person in image
    results = model.predict(db, conf=conf, iou=iou, save=save_result, save_txt=save_txt, hide_labels=True, hide_conf=True, classes=[0], stream=True)
    idx = 0
    for result in results:
        idx += 1
    model_path = model.export(format="onnx")
    return results, model_path

def read_frames_from_video(path="/home/tuantv/Desktop/hieu/classroom_selfcollect_dataset/Dell_2.mp4", step=1, save=True, save_path=None):
    video = cv2.VideoCapture(path)
    label = path.split("/")[-1].split(".")[0]
    try:
        if save_path is None:
            save_path = os.path.join("./", label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Create save path!")
    except OSError:
        print("Error")
    
    crt_frame = 0
    while(True):
        ret, frame = video.read()
        
        if ret:
            name = os.path.join(".", label, str(crt_frame)+".jpg")
            print('Captured ' + name)
            cv2.imwrite(name, frame)
            crt_frame += 1
            # break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    return save_path
    
if __name__ == "__main__":
    #video_path = "/home/tuantv/Desktop/hieu/classroom_selfcollect_dataset/WIN_20230807_14_48_17_Pro.mp4"
    # path = "/home/tuantv/Desktop/hieu/classroom/dataset/images/PartA_00097.jpg"

    # read_frames_from_video(path=video_path)
    db = "/home/tuantv/Desktop/hieu/classroom_selfcollect_dataset/"
    folders =[os.path.join(db, p) for p in os.listdir(db)[1:]] 
    for path in folders:
        detect_person(path=path)
    
    # db = "/home/tuantv/Desktop/hieu/detect_person/Dell_1"
    # detect_person(db, conf=0.01)