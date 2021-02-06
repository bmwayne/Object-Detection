import cv2
cocofile = 'coco.names' #list of object on which the pretrained model is trained on
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #pretrained deep learning model
weightspath = 'frozen_inference_graph.pb' #weights

classnames = []

with open(cocofile,'rt') as f:             #reading coco.names file
    classnames = f.read().split('\n')

video = cv2.VideoCapture(0) #setting the input camera or webcam in this case

#setting up the detection model
detect_model = cv2.dnn_DetectionModel(weightspath, configpath)
detect_model.setInputSize(320, 320)
detect_model.setInputScale(1.0/127.5)
detect_model.setInputMean((127.5,127.5,127.5))
detect_model.setInputSwapRB(True)

while True:
    check, frame = video.read()
    classIds, confs, bbox = detect_model.detect(frame, confThreshold=0.5)   #applying the cv2 detect method using the detection model
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0,255,0), thickness=2) #drawing a rectanle box around the detected object
            cv2.putText(frame, classnames[classId - 1].upper(),(box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,color=(0,255,0), thickness=2) #specifying the object
            cv2.putText(frame, str(round(confidence*100, 2)), (box[0] + 150, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        color=(0, 255, 0), thickness=2) #specifying the confidence level

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) == 27:
            break


video.release()
cv2.destroyAllWindows()