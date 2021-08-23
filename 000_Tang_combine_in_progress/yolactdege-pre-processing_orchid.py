from eval_ import *
from utils.logging_helper import setup_logger
from pypylon import pylon

#from google.colab.patches import cv2_imshow
import cv2
import time
parse_args(["--config=yolact_edge_config"])

from eval_ import args
from utils.tensorrt import convert_to_tensorrt

import time
setup_logger(logging_level=logging.INFO)
logger = logging.getLogger("yolact.eval")

#------------------------- Plantcv ---------------------------------#
# Import libraries
from plantcv import plantcv as pcv
import numpy as np
import math

# For plantcv
import subfunction
import PlantCV_main
import S2_findBranch
import S3_FindAngle

np.set_printoptions(threshold=np.inf)

args.image="result/043.jpg" 
args.mask_img = "mask/mask_1.png"
args.debug = "plot" #"print" or "plot"
args.mask_processed = "mask_processed/"

# Set debug to the global parameter
pcv.params.debug = args.debug
#-------------------------------------------------------------------#

args.trained_model = "./weights/yolact_edge_64_50000.pth"

args.yolact_transfer = True
args.disable_tensorrt = True
#args.use_fp16_tensorrt = True
#args.use_tensorrt_safe_mode = True
args.calib_images="./calib_images"

torch.set_default_tensor_type('torch.cuda.FloatTensor')

logger.info('Loading model...')
net = Yolact(training=False)
net.load_weights(args.trained_model, args=args)
net.eval()
logger.info('Model loaded.')

print("cfg.num_classes", cfg.num_classes)

convert_to_tensorrt(net, cfg, args, transform=BaseTransform())

net.detect.use_fast_nms = args.fast_nms
cfg.mask_proto_debug = args.mask_proto_debug

args.score_threshold = 0.1
args.top_k = 15

extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,
          "moving_statistics": None}

# cap = cv2.VideoCapture(0)
prevTime = 0

# define a video capture object
vid = cv2.VideoCapture(0)

Root_exist = False

while (True):
    # Capture the video frame
    # by frame
    img = cv2.imread("img_test/245.jpg")
    #img = cv2.resize(frame, (640, 480))
    # ret,img = cap.read()

    frame = torch.from_numpy(img).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    with torch.no_grad():
        preds = net(batch, extras=extras)["pred_outs"]

    #dets = preds[0]
    # for i in range(len(dets['score'])):
    #     if dets['score'][i]
    masks_out, classes, boxes, img_numpy = prep_display_orchid(preds, frame, None, None, undo_transform=False) # from eval import *

    cv2.imshow("original", img)
    cv2.imshow('processed', img_numpy)
    
    #----------- OPERATE ON SINGLE MASK  -----------------------------#
    list_branchPoints = []
    for i in range(len(masks_out)):
        # STEP 1: PLANTCV --> branchPoints, tipPoints
        if classes[i] == 0: 
            mask = (masks_out[i] * 255).byte().cpu().numpy()
            img = np.squeeze(mask, axis=2)
            #skeletonPoints, branchPoints, tipPoints, branch_plot = PlantCV_main.GetData(maskPath)
            skeletonPoints, branchPoints, tipPoints, branch_plot = PlantCV_main.Read_img(img)
    
            print("tipPoints = ", tipPoints)
            print("old branchPoints = ", branchPoints)
            all_2ends = subfunction.find_2ends_list(skeletonPoints)

            branchPoints = subfunction.clearBranchPoint(branchPoints)
            print("NEW branchPoints = ", branchPoints)
    
            branchPoints = branchPoints + tipPoints
            print("branchPoints + tipPoints = ", branchPoints)
            
            list_branchPoints.append(branchPoints)
        else:
            # Root exist
            Root_exist = True
            img_numpy_test = (masks_out[i] * 255).byte().cpu().numpy()
            cv2.imwrite("mask/root_{}.png".format(i), img_numpy_test)
            # center of root
            x,y = boxes[i][0] + int((boxes[i][2] - boxes[i][0])/2), boxes[i][1] + int((boxes[i][3] - boxes[i][1])/2)
            centerPointOfDark = [x,y]
            print("x, y = ", x, y)
    print("list_branchPoints = ", list_branchPoints)
    #----------- OPERATE ON SINGLE MASK  -----------------------------#
    # If not Root --> find the group of closest point using KNN
    #if (Root_exist == False):
        # KNN to find a group of n closest points
    print("len(list_branchPoints) = ", len(list_branchPoints))
    dis_list = []
    index = []
    #dis_list, index = S2_findBranch.A_root_proposed(list_branchPoints)
    #print("dis_list = {}, index = {}".format(dis_list, index ))
    print("list_branchPoints[0] = ", list_branchPoints[0])
    print("list_branchPoints[0][0] = ", list_branchPoints[0][0])
    print("list_branchPoints[0][1] = ", list_branchPoints[0][1])
    '''
    #----------- OPENPLANT OPERATION     -----------------------------#                  
# STEP 1: PLANTCV --> branchPoints, tipPoints, ... --> zero_point  
    zero_point = subfunction.nearestPoint(branchPoints, centerPointOfDark)
    print("zero_point = ", zero_point)
    
    if zero_point in tipPoints:
        tipPoints.remove(zero_point)       
    
# STEP 2: Find ONE longest branch
    # 2.1 Find All Branch
    all_branches = []
    zero_point_ID = -100
    S2_findBranch.f(zero_point, zero_point_ID, all_2ends, branchPoints, tipPoints, [zero_point], all_branches)

    # 2.2 Get longest branch
    print("[i for i in range(len(all_branches))]" ,[i for i in range(len(all_branches))])
    all_branch_lengths = S2_findBranch.LengthOfBranches([i for i in range(len(all_branches))], all_branches)
    print("all_branch_lengths = ", all_branch_lengths)
    id_longestBud = all_branch_lengths.index(max(all_branch_lengths))
    longestBud = all_branches[id_longestBud]
    print("longestBud = ", longestBud)
    budLength = all_branch_lengths[id_longestBud]
    print("budLength = ", budLength)

# STEP 3: Angle of Branch
    x = [longestBud[1][0]]
    y = [longestBud[1][1]]
    dist = [math.dist(longestBud[1],zero_point)]
    angle_out = S3_FindAngle.angle_cal(x, y, zero_point, dist)
    '''
    #----------- DRAW SINGLE MASK -----------------------------------#
    
    # Show FPS
    curTime = time.time()
    if (curTime - prevTime) != 0:
        fps = 1/(curTime - prevTime)
    prevTime = curTime
    print(fps)

    k = cv2.waitKey(1)
    if k == 27:
        break    
    time.sleep(2)
    
    
    
# Releasing the resource    
cv2.destroyAllWindows()


