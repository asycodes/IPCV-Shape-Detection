import numpy as np
import cv2
import os
import sys
import argparse
import json
from hough.sobel import sobel
from hough.hough_circle import hough_circle_with_orientation

"""
1) Create no entry detector model and find all of them, change parameters of the model to maximise accuracy
2) non_maximum_suppression to get rid of redundant detection
3) ground truth to get performance scores (using iou with the ground truth)

"""

parser = argparse.ArgumentParser(description="no entry")
parser.add_argument("-name", "-n", type=str, default="No_entry/NoEntry0.bmp")
args = parser.parse_args()

cascade_name = "NoEntryCascade/cascade.xml"
ground_truth_bounding_boxes = []
pred_bounding_boxes = []


def detectAndDisplay(frame):
    global pred_bounding_boxes
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    cv2.imwrite("preprocessed.jpg", frame_gray)
    noEntry = model.detectMultiScale(
        frame_gray,
        scaleFactor=1.05,
        minNeighbors=1,
        flags=0,
        minSize=(3, 3),
        maxSize=(400, 400),
    )
    sobel(imageName)
    min_radius = 10
    max_radius = 100
    threshold_hough = 18
    pred_bounding_boxes = hough_circle_with_orientation(
        imageName, min_radius, max_radius, threshold_hough
    )
    readGroundtruth(frame, imageName.split("/")[1])

    ### uncomment to see all faces before filtering out
    for i in range(0, len(noEntry)):
        start_point = (noEntry[i][0], noEntry[i][1])
        end_point = (noEntry[i][0] + noEntry[i][2], noEntry[i][1] + noEntry[i][3])
        pred_bounding_boxes.append(
            (start_point, end_point, noEntry[i][2], noEntry[i][3])
        )
        """colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)"""
    print(len(pred_bounding_boxes))
    update_bounding_boxes()
    print(len(pred_bounding_boxes))

    for noEntry in pred_bounding_boxes:
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, noEntry[0], noEntry[1], colour, thickness)
    success = evaluate_iou()
    print("SUCCESS: ", success)
    tpr_result = tpr(success)
    print("TPR: ", tpr_result)
    f1_result = f1(success)
    print("F1: ", f1_result)


def readGroundtruth(frame, imageName, filename="groundtruth.json"):

    with open(filename, "r") as f:
        data = json.load(f)

    for item in data:
        file_upload = item["file_upload"]
        if file_upload.endswith(imageName):
            annotations = item["annotations"]
            for annotation in annotations:
                for result in annotation["result"]:

                    x = int(result["value"]["x"] * result["original_width"] / 100)
                    y = int(result["value"]["y"] * result["original_height"] / 100)
                    width = int(
                        result["value"]["width"] * result["original_width"] / 100
                    )
                    height = int(
                        result["value"]["height"] * result["original_height"] / 100
                    )
                    start_point = (x, y)
                    end_point = (x + width, y + height)
                    ground_truth_bounding_boxes.append(
                        (start_point, end_point, width, height)
                    )

                    colour = (0, 0, 255)
                    thickness = 2
                    frame = cv2.rectangle(
                        frame, start_point, end_point, colour, thickness
                    )


def iou(pred, truth):
    start_x = max(pred[0][0], truth[0][0])
    start_y = max(pred[0][1], truth[0][1])
    end_x = min(pred[1][0], truth[1][0])
    end_y = min(pred[1][1], truth[1][1])

    inter_width = max(0, end_x - start_x)
    inter_height = max(0, end_y - start_y)
    area_inters = inter_width * inter_height

    area_pred = (pred[1][0] - pred[0][0]) * (pred[1][1] - pred[0][1])
    area_truth = (truth[1][0] - truth[0][0]) * (truth[1][1] - truth[0][1])

    area_union = area_pred + area_truth - area_inters

    if area_union == 0:
        return 0
    iou_score = area_inters / area_union
    return iou_score


def evaluate_iou():
    success = 0
    for pred in pred_bounding_boxes:
        for truth in ground_truth_bounding_boxes:
            print(iou(pred, truth))
            if iou(pred, truth) > 0.45:
                success += 1
                break
    return success


def tpr(success):
    return success / len(ground_truth_bounding_boxes)


def f1(success):
    if len(pred_bounding_boxes) == 0:
        return 0
    # 2*[(precision*recall)/(precision+recall)]
    precision = success / len(pred_bounding_boxes)
    # false negative is truth - pred
    recall = success / (success + len(ground_truth_bounding_boxes) - success)
    if precision == 0 and recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))


## extra: NMS - to reduce redundant repeating boundary boxes
def non_maximum_suppression(bboxes, overlap_thresh=0.5):
    if len(bboxes) == 0:
        return []

    bboxes = np.array(bboxes, dtype=object)

    x1 = np.array([bbox[0][0] for bbox in bboxes])
    y1 = np.array([bbox[0][1] for bbox in bboxes])
    x2 = np.array([bbox[1][0] for bbox in bboxes])
    y2 = np.array([bbox[1][1] for bbox in bboxes])
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(area)[::-1]

    picked_boxes = []

    while len(idxs) > 0:
        current = idxs[0]
        picked_boxes.append(bboxes[current])
        remaining_boxes = idxs[1:]
        ious = np.array([iou(bboxes[current], bboxes[i]) for i in remaining_boxes])
        print(ious)
        idxs = remaining_boxes[ious <= overlap_thresh]

    return picked_boxes


def update_bounding_boxes():
    global pred_bounding_boxes
    pred_bounding_boxes = non_maximum_suppression(
        pred_bounding_boxes, overlap_thresh=0.2
    )


def save_detection(frame, imageName):
    output_folder = "detected_img"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.basename(imageName)

    output_path = os.path.join(output_folder, base_name)
    cv2.imwrite(output_path, frame)
    print(f"Saved detection to {output_path}")


# ==== MAIN ==============================================

imageName = args.name


# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print("No such file")
    sys.exit(1)

# 1. Read Input Image
frame = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(frame) is np.ndarray):
    print("Not image data")
    sys.exit(1)


# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier()
if not model.load(
    cascade_name
):  # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    print("--(!)Error loading cascade model")
    exit(0)


# 3. Detect Faces and Display Result
detectAndDisplay(frame)

# 4. Save Result Image
save_detection(frame, imageName)
