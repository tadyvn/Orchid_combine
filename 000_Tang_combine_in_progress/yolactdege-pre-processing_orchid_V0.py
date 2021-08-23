# NOTE: the first version to test the combination 
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


args.trained_model = "./weights/yolact_edge_51_40000.pth"

args.yolact_transfer = True
args.disable_tensorrt = True
#args.use_fp16_tensorrt = True
#args.use_tensorrt_safe_mode = True
args.calib_images="./calib_images"

args.image="result/043.jpg" 
args.mask_img = "mask/mask_1.png"
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

while (True):
    # Capture the video frame
    # by frame
    img = cv2.imread("img_test/000.jpg")
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

    #----------- DRAW SINGLE MASK -----------------------------------#
    for i in range(len(masks_out)):
        if classes[i] == 0: 
            img_numpy_test = (masks_out[i] * 255).byte().cpu().numpy()
            cv2.imwrite("mask/mask_{}.png".format(i), img_numpy_test) 
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


