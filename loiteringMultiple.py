import cv2
from smartsight_services import loiteringFaceCover
from time import sleep
from smartsight_services.MotionDetectionManager import MotionDetectionManager
from PIL import Image
import io
import requests
from datetime import datetime
import sys
from smartsight_services.ApplianceSetting import ApplianceSetting
from pytz import timezone
import json
import datetime as dt
import time
from smartsight_services.loitringDetectionHelper import loiteringDetector
from smartsight_services.person_identifier import person_identifier
from smartsight_services.person_detector import person_detector, SSD_person_detector
from smartsight_services.IntelCalssifyInferencer import IntelCalssifyInferencer
from smartsight_services.person_db_loitering import person_db
import numpy as np
from vidutils import WebCamVideoStream
from smartsight_services.v4_detection import detection


def upload_sd_new(check_in_time, camera_obj, image):
    try:

        request_url = camera_obj.url + "/api/alert/compliance/"
        my_headers = {'Authorization': 'Token ' + camera_obj.token}
        files = {"images": []}

        pil_im = Image.fromarray(image)
        temp = io.BytesIO()
        pil_im.save(temp, format='JPEG')
        temp.seek(0)
        files["images"] = temp
        data = {'check_in': check_in_time.strftime('%Y-%m-%d %H:%M:%S') + "+00:00", 'sector': camera_obj.sector,
                'type': 'CROWD_ALERT',
                'description': "{\"type\":\"crowd\"}",
                'setting_id': camera_obj.setting.setting_id}

        response = requests.post(request_url, headers=my_headers, data=data, files=files)
        print(response.text)

    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)

def upload_sd_new_FaceCover(check_in_time, camera_obj, image,time):
    try:
        frmt_date = check_in_time.strftime('%Y-%m-%d %H:%M:%S') + "+00:00"
        request_url = camera_obj.url + "/api/alert/compliance/"
        my_headers = {'Authorization': 'Token ' + camera_obj.token}
        files = {"images": []}

        pil_im = Image.fromarray(image)
        temp = io.BytesIO()
        pil_im.save(temp, format='JPEG')
        temp.seek(0)
        files["images"] = temp
        data = {'check_in': check_in_time.strftime('%Y-%m-%d %H:%M:%S') + "+00:00", 'sector': camera_obj.sector,
                'type': 'HELMET_ALERT',
                'description': "{\"duration\":\""+frmt_date+"\"}",
                'setting_id': camera_obj.setting.setting_id}


        response = requests.post(request_url, headers=my_headers, data=data, files=files)
        print(response.text)

    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)

def upload_sd_new_loit(check_in_time, camera_obj, image,time):
    try:
        frmt_date = dt.datetime.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S') + "+00:00"
        request_url = camera_obj.url + "/api/alert/compliance/"
        my_headers = {'Authorization': 'Token ' + camera_obj.token}
        files = {"images": []}

        pil_im = Image.fromarray(image)
        temp = io.BytesIO()
        pil_im.save(temp, format='JPEG')
        temp.seek(0)
        files["images"] = temp
        data = {'check_in': check_in_time.strftime('%Y-%m-%d %H:%M:%S') + "+00:00", 'sector': camera_obj.sector,
                'type': 'PERSON_LOITER',
                'description': "{\"duration\":\""+frmt_date+"\"}",
                'setting_id': camera_obj.setting.setting_id}


        response = requests.post(request_url, headers=my_headers, data=data, files=files)
        print(response.text)

    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)



 
          

def update_camera_preview(camera, image):
    try:
        headers = {
            'Authorization': 'Token ' + camera.token}
        request_url = camera.url + "/api/camera_update/"
        files = {"preview_image": []}
        pil_im = Image.fromarray(image)
        temp = io.BytesIO()
        pil_im.save(temp, format='JPEG')
        temp.seek(0)
        files["preview_image"] = temp
        data = {'preview_time': datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S') + "+00:00", 'camera_id': camera.camera_id}
        response = requests.post(
            request_url, headers=headers, data=data, files=files)
        print(response.text)
        if response.status_code == 201:
            print(response.text)
            return response.text
        else:
            return ""
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)



def update_camera_time(camera):
    try:
        headers = {'content-type': 'application/json',
                   'Authorization': 'Token ' + camera.token}
        request_url = camera.url + "/api/camera_update/"
        data = {'preview_time': datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S') + "+00:00", 'camera_id': camera.camera_id}
        response = requests.post(
            request_url, headers=headers, data=json.dumps(data, indent=4))

        print(response.text)
        if response.status_code == 201:
            print(response.text)
            return response.text
        else:
            return ""

    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)

def check_is_in(xmin,xmax,ymin,ymax,roi_xmin,roi_xmax,roi_ymin,roi_ymax):
    if roi_xmin < xmin and xmax < roi_xmax and roi_ymin < ymin and ymax < roi_ymax:
        return True
    else:
        return False

def start(camera,display):
    person_detects = ["models/pdr_013.bin","models/pdr_013.xml"]
    personDetector = person_detector(person_detects[0],
                                          person_detects[1])
    # identifier class call
    pd_vector = ["models/per_ident_0288.bin",
                 "models/per_ident_0288.xml"]
    personVector = person_identifier(pd_vector[0],
                                              pd_vector[1])

    # ppe classifier
    cover_face = ["models/coverFace.bin","models/coverFace.xml"]
    faceClassifier = IntelCalssifyInferencer(cover_face[1], cover_face[0],["covered","uncovered"])




    # print(type(camera))
    for key,value in camera.items():
        print(key,value["camera"].video_url,type(camera[key]))
        try:
            print(time.time())
            # vs = cv2.VideoCapture(value["camera"].video_url)
            value["CamFeed"] = WebCamVideoStream(src=value["camera"].video_url).start()

            # for i in range (3):
            frame = value["CamFeed"].read()
            print(time.time())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            value["camera"].update_preview(rgb)
            (height, width) = frame.shape[:2]
            x_param = width / 1280
            y_param = height / 720
            X1 = int(value["camera"].setting.rect.x * x_param)
            X2 = int((value["camera"].setting.rect.x + value["camera"].setting.rect.width) * x_param)
            Y1 = int(value["camera"].setting.rect.y * y_param)
            Y2 = int((value["camera"].setting.rect.y + value["camera"].setting.rect.height) * y_param)
            crowdTimeLimit = 2  #seconds
            value["totalFrames"] = 0
            value["ltrTRack"] = loiteringDetector(value["camera"].setting.allRoi, value["camera"].setting.loiter_time, crowdTimeLimit)
            value["person_database"] = person_db(60)
            value["MotionDetector"] = MotionDetectionManager(value["camera"].setting.minMotionSize, frame)
        except Exception as e:
            print("NotWorking",e)


    while True:
        print("Time to cover all:",time.time())
        for key,value in camera.items():
            # print(key,value["camera"].video_url,type(camera[key]))
            # check if current_time - LastProcessTime > waitTime
            current_time = time.time()
            # print(current_time)
            coveredFace = []

            if  value["waitTime"]==None or current_time - value["LastProcessTime"] > value["waitTime"] :

                try:
                    # vs = cv2.VideoCapture(value["camera"].video_url)
                    # for i in range (3):
                    #     res, frame = vs.read()

                    frame = value["CamFeed"].read()
                    frame_copy = frame.copy()
                    (height, width) = frame.shape[:2]
                    
                    x_param = width / 1280
                    y_param = height / 720
                    X1 = int(value["camera"].setting.rect.x * x_param)
                    X2 = int((value["camera"].setting.rect.x + value["camera"].setting.rect.width) * x_param)
                    Y1 = int(value["camera"].setting.rect.y * y_param)
                    Y2 = int((value["camera"].setting.rect.y + value["camera"].setting.rect.height) * y_param)
                    # value["MotionDetector"] = MotionDetectionManager(value["camera"].setting.minMotionSize, frame)


                    motion_detection_flag = value["MotionDetector"].process(frame)
                    print("motion_detection_flag: ",motion_detection_flag)
                    if not motion_detection_flag:
                        value["waitTime"] = 1
                        value["LastProcessTime"] = current_time
                        value["LastProcessFrame"] = frame
                        continue
                    person_cords = personDetector.infer(frame, width, height,
                                                            value["camera"].setting.confidence)  # detection of persons will return coords of detected persons
                    # print(len(person_cords))
                    frame_copy = frame.copy()
                    blankImage = np.zeros(shape=(height, 2 * width, 3), dtype=np.uint8)
                    blankImage[:, :width, :] = frame

                    ltrCords = []
                    crowds = []
                    coveredFace = []
                    alertType = None
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # value["camera"].update_preview(rgb)

                    if len(person_cords) > 0:
                        for xmin, xmax, ymin, ymax, label in person_cords:
                            inside = value["ltrTRack"].check_roi_Lower(xmin, xmax, ymin, ymax)
                            #if not inside:
                                # print("not inside")
                            #    continue

                            # print("inside", len(person_cords))



                            # identification for each crop image
                            # cropping person from frame
                            iframe = frame[abs(ymin):abs(ymax), abs(xmin):abs(xmax)]
                            vector_result = personVector.identify(iframe)

                            is_covered = "False" #faceClassifier.infer(iframe, iframe.shape[1], iframe.shape[0])
                            person, person_id = value["person_database"].get_person([vector_result], 0.7, None, frame_copy)

                            if not person["faceCovered"] and is_covered == "covered" :
                                # alert = ["No ppe"]
                                coveredFace.append(True)
                                value["person_database"].cover_alerted(person_id)

                            # check if he is seen previously in the roi
                            isLoitering = person["loitering"]
                            # if debug:
                            #     print(isLoitering)

                            if not isLoitering:
                                loitring = False
                                entryTime = person["entryTime"]

                                # calculating centoid and loitring detection
                                isLoitering = value["ltrTRack"].centroidDetector(
                                    xmin, ymin, xmax, ymax, entryTime)
                                isLoitering = isLoitering[0]

                                # isLoitering = self.ltrTRack.check_loitering_time(entryTime)

                            # already detected as loitering
                            if isLoitering:
                                loitring = True

                                # append to ltrlist
                                if not person["alerted"]:
                                    # clubbed = cv2.hconcat([frame, person["image"]])
                                    if person["image"] is not None:
                                        blankImage[:, width:, :] = person["image"]
                                        ltrCords.append({"cords": [person_id, xmin, ymin, xmax, ymax], "image": blankImage,
                                                        "time": person["entryTime"]})
                                        # value["person_database"].change_alerted(person_id)
                            stayTimeExceeded = value["ltrTRack"].check_crowd_time(person["entryTime"])

                            # if not person["crowd"]:
                            crowds.append([person_id, xmin, ymin, xmax, ymax])
                            # value["person_database"].crowd_alerted(person_id)
                                # self.alerted.append[person_id]

                            value["person_database"].person_update(
                                    person_id, [vector_result], loitring)
                    # value["person_database"].delete_person_image()


                    if len(ltrCords) > 0:
                        alertType = "Loitering"

                        # print(len(ltrCords))
                        
                        image = cv2.resize(ltrCords[0]["image"], (1280, 720))
                        upload_sd_new_loit(datetime.utcnow(
                        ), value["camera"], cv2.cvtColor(image, cv2.COLOR_BGR2RGB),time)
                        value["waitTime"] = 105
                        value["LastProcessTime"] = current_time
                        value["LastProcessFrame"] = frame

                        # print( ltrCords, alertType, type(ltrCords[0]["image"]), ltrCords[0]["time"])
                        # return ltrCords, alertType, ltrCords[0]["image"], ltrCords[0]["time"]
                    elif len(crowds) >= value["camera"].setting.max_occupancy:
                        alertType = "Crowd"
                        # if totalFrames - alertedFrame >= 4000 or totalFrames < 4000:
                        upload_sd_new(datetime.utcnow(
                        ), value["camera"], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        value["camera"].update_preview(rgb)

                        value["waitTime"] = 105
                        value["LastProcessTime"] = current_time
                        value["LastProcessFrame"] = frame

                        # alertedFrame = totalFrames
                    elif len(coveredFace)>0:
                        # alertType = "CoveredFace"
                        upload_sd_new_FaceCover(datetime.utcnow(
                            ), camera, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time)
                        value["waitTime"] = 105
                        value["LastProcessTime"] = current_time
                        value["LastProcessFrame"] = frame
                    #         # cv2.imshow("iframe", iframe)
                    #         # cv2.waitKey(0)
                    else:
                        # alert = []
                        # upload_sd_new(datetime.utcnow(
                        # ), camera, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        value["waitTime"] = 30
                        value["LastProcessTime"] = current_time
                        value["LastProcessFrame"] = frame


                    if display=="True":
                        cv2.imshow("Frame", frame)
                        cv2.waitKey(1)
                        # vs.release()
                        # key = cv2.waitKey(1) & 0xFF
                        # if key == ord("q"):
                        #     break
                except Exception as e:
                    print("NotWorkinghere also",e)

    # return

    # vs.release()
    cv2.destroyAllWindows()



#starting here

import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import concurrent
from multiprocessing import Pool
import multiprocessing
import logging

logging.basicConfig(filename='appliance.log',format='%(asctime)s %(message)s',  level=logging.DEBUG)
myQ = Queue()


class FrameObj():
    def __init__(self, frame, camera):
        self.frame = frame
        self.camera = camera
     


 


def process_frame(frame, cameraDictobj, personDetector):
    value = cameraDictobj
    current_time = time.time()

    #1b162d7d-1a0e-4ebc-8fb9-3c8fbc219e24

  
    if  value["waitTime"]==None or current_time - value["LastProcessTime"] > value["waitTime"] :

        try:
            
            height = None
            width = None
            X1 = None
            X2 =None
            Y1 = None
            Y2 = None

            if value["coordinates"] is None:
                height = None
                width = None

                (height, width) = frame.shape[:2]
                
                # x_param = width / 1280
                # y_param = height / 720
                # X1 = int(value["camera"].setting.rect.x * x_param)
                # X2 = int((value["camera"].setting.rect.x + value["camera"].setting.rect.width) * x_param)
                # Y1 = int(value["camera"].setting.rect.y * y_param)
                # Y2 = int((value["camera"].setting.rect.y + value["camera"].setting.rect.height) * y_param)
                value["coordinates"] = (height, width, X1,X2,Y1,Y2)
                crowdTimeLimit = 2 
                value["totalFrames"] = 0
                #value["ltrTRack"] = loiteringDetector(value["camera"].setting.allRoi, value["camera"].setting.loiter_time, crowdTimeLimit)
                value["ltrTRack"] = loiteringDetector(value["camera"].setting.allRoi, 1, crowdTimeLimit)
                value["person_database"] = person_db(60)
                value["MotionDetector"] = MotionDetectionManager(value["camera"].setting.minMotionSize, frame)


            else:
                height, width, X1,X2,Y1,Y2 = value["coordinates"]
                (height, width) = frame.shape[:2]
                
            # value["MotionDetector"] = MotionDetectionManager(value["camera"].setting.minMotionSize, frame)


            #motion_detection_flag = value["MotionDetector"].process(frame)
            # print("motion_detection_flag: ",motion_detection_flag)
            #if not motion_detection_flag:
            #    value["waitTime"] = 1
            #    value["LastProcessTime"] = current_time
            #    value["LastProcessFrame"] = frame
            #    return ""
            #person_cords = personDetector.infer(frame, width, height,s
            #                                        value["camera"].setting.confidence)  # detection of persons will return coords of detected persons
            # person_cords = personDetector.infer(frame, width, height,.6)
            person_cords = personDetector.infer(frame,.35)

            # print(len(person_cords))
            frame_copy = frame.copy()
            blankImage = np.zeros(shape=(height, 2 * width, 3), dtype=np.uint8)
            blankImage[:, :width, :] = frame

            ltrCords = []
            crowds = []
            coveredFace = []
            alertType = None
            #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # value["camera"].update_preview(rgb)

            if len(person_cords) > 0:
                # print(person_cords)
                # for xmin, xmax, ymin, ymax, label in person_cords:
                for obj in person_cords:
                    xmin, xmax, ymin, ymax, label = obj["xmin"], obj["xmax"],obj["ymin"],obj["ymax"],obj["class_id"]
                    #inside = value["ltrTRack"].check_roi_Lower(xmin, xmax, ymin, ymax)
                    # avoiding the inside outside 
                    #if not inside:
                        # print("not inside")
                    #    continue
                    # print(label,type(label))
                    if label==0:
                        print("got person ", len(person_cords))
                        # identification for each crop image
                        # cropping person from frame
                        iframe = frame[abs(ymin):abs(ymax), abs(xmin):abs(xmax)]
                        vector_result = personVector.identify(iframe)
                        is_covered  = "False"
                        #is_covered = faceClassifier.infer(iframe, iframe.shape[1], iframe.shape[0])
                        person, person_id = value["person_database"].get_person([vector_result], 0.8, None, frame_copy)

                        if not person["faceCovered"] and is_covered == "covered" :
                            # alert = ["No ppe"]
                            coveredFace.append(True)
                            value["person_database"].cover_alerted(person_id)

                        # check if he is seen previously in the roi
                        isLoitering = person["loitering"]
                        # if debug:
                        #     print(isLoitering)

                        if not isLoitering:
                            loitring = False
                            entryTime = person["entryTime"]

                            # calculating centoid and loitring detection
                            isLoitering = value["ltrTRack"].centroidDetector(
                                xmin, ymin, xmax, ymax, entryTime)
                            if len(isLoitering)>0:
                                isLoitering = isLoitering[0]

                            # isLoitering = self.ltrTRack.check_loitering_time(entryTime)

                        # already detected as loitering
                        if isLoitering:
                            loitring = True

                            # append to ltrlist
                            if not person["alerted"]:
                                # clubbed = cv2.hconcat([frame, person["image"]])
                                if person["image"] is not None:
                                    print(value["camera"].camera_id)
                                    blankImage[:, width:, :] = person["image"]
                                    ltrCords.append({"cords": [person_id, xmin, ymin, xmax, ymax], "image": blankImage,
                                                    "time": person["entryTime"]})
                                    # value["person_database"].change_alerted(person_id)
                        stayTimeExceeded = value["ltrTRack"].check_crowd_time(person["entryTime"])

                        # if not person["crowd"]:
                        crowds.append([person_id, xmin, ymin, xmax, ymax])
                        # value["person_database"].crowd_alerted(person_id)
                            # self.alerted.append[person_id]

                        value["person_database"].person_update(
                                person_id, [vector_result], loitring)
                # value["person_database"].delete_person_image()


            if len(ltrCords) > 0:
                alertType = "Loitering"

                # print(len(ltrCords))
                print("Loiter alert")
                image = cv2.resize(ltrCords[0]["image"], (1280, 720))
              
                upload_sd_new_loit(datetime.utcnow(), value["camera"], cv2.cvtColor(image, cv2.COLOR_BGR2RGB),1)
                print("Loiter loiter alert sent")
                value["waitTime"] = 105
                value["LastProcessTime"] = current_time
                value["LastProcessFrame"] = frame

                # print( ltrCords, alertType, type(ltrCords[0]["image"]), ltrCords[0]["time"])
                # return ltrCords, alertType, ltrCords[0]["image"], ltrCords[0]["time"]
            #elif len(crowds) >= value["camera"].setting.max_occupancy:
            elif len(crowds) >= 2:
                alertType = "Crowd"
                # if totalFrames - alertedFrame >= 4000 or totalFrames < 4000:
                upload_sd_new(datetime.utcnow(
                ), value["camera"], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                #value["camera"].update_preview(rgb)

                value["waitTime"] = 105
                value["LastProcessTime"] = current_time
                value["LastProcessFrame"] = frame

                # alertedFrame = totalFrames
            elif len(coveredFace)>0:
                # alertType = "CoveredFace"
                upload_sd_new_FaceCover(datetime.utcnow(
                    ), camera, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time)
                value["waitTime"] = 105
                value["LastProcessTime"] = current_time
                value["LastProcessFrame"] = frame
            #         # cv2.imshow("iframe", iframe)
            #         # cv2.waitKey(0)
            else:
                # alert = []
                # upload_sd_new(datetime.utcnow(
                # ), camera, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                value["waitTime"] = 1
                value["LastProcessTime"] = current_time
                value["LastProcessFrame"] = frame
         
                # vs.release()
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord("q"):
                #     break
        except Exception as e:
            print("Exception in process",e)
            logging.debug("Process exception {0} {1}".format(value["camera"].camera_id ,e))




def convert_to_tcp(url):
    return url.replace("rtsp","rtspt")


def get_frame(camera):
    # vs = cv2.VideoCapture(convert_to_tcp(camera.video_url))
    vs = cv2.VideoCapture(camera.video_url)

    start_time = datetime.now()
    working = 0
    not_working = 0
    count = 0

    skip = False

    
    if not skip:
        while True:
            # try:
                res, frame = vs.read()
                count = count + 1
                if res:
                    working = working + 1
                    res, frame = vs.read()
                    # print("Got Frame at {0}  ".format(datetime.now() - start_time))
                    vs.release()
                    return frame, camera.camera_id
                    break
                if count >  2:
                    #print("Not got Frame at {0} id-{1} ".format(datetime.now() - start_time, camera.camera_id))
                    not_working = not_working + 1
                    vs.release()
                    return None, camera.camera_id

            # except:
            #     not_working = not_working + 1
            #     logging.debug("Not got Frame exception at {0} {1}".format(datetime.now() - start_time), camera.camera_id)
            #     #logging.debug("{0} Unexpected error:".format(camera.camera_id), sys.exc_info())
            #     vs.release()
            #     return None, camera.camera_id
    vs.release()
    return None,camera.camera_id
                

import os

                    
def save_frame(frame, camera_id):
    if not os.path.exists(camera_id):
        os.makedirs(camera_id)
    frame = cv2.resize(frame,  (640, 360))
    cv2.imwrite("{0}/{1}.jpg".format(camera_id,datetime.now().strftime("%d-%m-%Y-%H%M%S")), frame)






if __name__ == "__main__":
    

    
     with open("config.json") as file:
            
            
            data = json.load(file)
            app = ApplianceSetting(data["username"], data["password"],
                                                    data["url"], data["appliance_key"])

            # #42c3afd4-f751-47ff-a258-65ccad9f4160
            #app2 = ApplianceSetting(data["username"], data["password"],
            #                                         data["url"], "42c3afd4-f751-47ff-a258-65ccad9f4160")

            # #705ea5dc-6697-4f03-b51d-ef9f58962363
            #app3 = ApplianceSetting(data["username"], data["password"],
            #                                         data["url"], "705ea5dc-6697-4f03-b51d-ef9f58962363")
            print(datetime.now().strftime("%d-%m-%Y%H%M%S"))
            camerasDict = {}
            camera_list = []
            for camera in app.cameras:
                camera_list.append(camera)
                print(convert_to_tcp(camera.video_url))

                camerasDict[camera.camera_id] = {"camera":camera, "MotionState" : None, "waitTime": None, "LastProcessTime" : None, "LastProcessFrame": None, "MotionDetector": None, "person_database":None, "coordinates":None, "offline_status":None, "offline_check_time":None}
         
            # for camera in app2.cameras:
            #      camera_list.append(camera)
            #      camerasDict[camera.camera_id] = {"camera":camera, "MotionState" : None, "waitTime": None, "LastProcessTime" : None, "LastProcessFrame": None, "MotionDetector": None, "person_database":None, "coordinates":None, "offline_status":None, "offline_check_time":None}

            
            # for camera in app3.cameras:
            #      camera_list.append(camera)
            #      camerasDict[camera.camera_id] = {"camera":camera, "MotionState" : None, "waitTime": None, "LastProcessTime" : None, "LastProcessFrame": None, "MotionDetector": None, "person_database":None, "coordinates":None, "offline_status":None, "offline_check_time":None}


            print("{0} Cameras Loaded".format(len(camera_list)))
            print(camerasDict.keys())

            # person_detects = ["models/pvbd-0078.bin","models/pvbd-0078.xml"]
            # person_detects = ["models/ssd_mobilenet_v2.bin","models/ssd_mobilenet_v2.xml"]

            # personDetector = person_detector(person_detects[0],
            #                               person_detects[1])
            # personDetector = SSD_person_detector(person_detects[0],
            #                               person_detects[1])
            personDetector = detection("models/frozen_darknet_yolov4_model.xml","models/yolov4IR/frozen_darknet_yolov4_model.bin")

            # identifier class call
            pd_vector = ["models/per_ident_0288.bin",
                        "models/per_ident_0288.xml"]
            personVector = person_identifier(pd_vector[0],
                                                    pd_vector[1])

            # ppe classifier
            #cover_face = ["models/coverFace.bin","models/coverFace.xml"]
            #faceClassifier = IntelCalssifyInferencer(cover_face[1], cover_face[0],["covered","uncovered"])

            print("Models Loaded")
            
            i = 0
            working = 0
            not_working = 0
            threads = []
            
            multiprocessing.set_start_method('spawn')
            while True:
                # try:

                    over_all_start = datetime.now()
                    
                    pool = Pool(maxtasksperchild=1)
                    # pool = Pool(maxtasksperchild=1)
                    a = pool.imap( get_frame, camera_list)
                    pool.close()
                    pool.join()
                    print("final score {0} working {1} not working {2} total time {3}".format(len(camera_list), myQ.qsize(), 0 , datetime.now()-over_all_start))
                    pool.terminate()
                    cameras_offline = []
                    for cam, camera_id in a:
                        #print(camera_id)
                        #print(camerasDict[camera_id]["camera"].camera_id)
                        if cam is None:
                            cameras_offline.append(camera_id)
                            for cam in camera_list:
                                if cam.camera_id == cam:
                                    cam.skip = True
                            camerasDict[camera_id]["offline_status"] = True
                            camerasDict[camera_id]["offline_check_time"] = datetime.now()
                        else:
                            #print(camerasDict[camera_id])
                            process_frame(cam,camerasDict[camera_id],personDetector)
                            save_frame(cam, camera_id)

                    
                    for cam in cameras_offline:
                        logging.debug("OFFLINE "+ cam)
                    app.update_bulk_camera(cameras_offline)
                    print("total time {0}".format( datetime.now()-over_all_start))
                  


                # except:
                #     print("Unexpected error:", sys.exc_info())


                        
               


           

            # with concurrent.futures.ThreadPoolExecutor(8) as executor:
            #     executor.map(get_frame, app.cameras)

            
            exit(0)


            
            for camera in app.cameras:
                i = i + 1
                print(camera.setting.mode)
                start_time = datetime.now()
                
                if camera.setting.mode == "PPTR" :
                    #w, n_w, diff, frame = get_frame(camera)
                    threads.append(threading.Thread(target=get_frame, args=(camera,)))
            
            for t in threads:
                t.start()

            for t in threads:
                t.join()


                    #working = working+ w
                    #not_working = not_working + n_w
            
                
            print("final score {0} working {1} not working {2} total time {3}".format(i, myQ.qsize(), 0 , datetime.now()-over_all_start))





                        

                    # if camera.camera_id != "" :
                    #camerasDict[f"{i}"] = {"camera":camera, "MotionState" : None, "waitTime": None, "LastProcessTime" : None, "LastProcessFrame": None, "MotionDetector": None, "person_database":None}
                    # print(camera.__dict__)
                    #print(camerasDict[str(i)]["camera"].name)
                    #i+=1
            #start(camerasDict,data["display"])

# def cameraDict,data["choose"]\
#   start(camera.____dict___)
