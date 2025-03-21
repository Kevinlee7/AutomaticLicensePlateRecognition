import cv2
import torch
import numpy as np
import time
from lpr_recognizer import LPRRecognizer
from PIL import Image, ImageDraw, ImageFont

def open_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"摄像头 {index} 打开成功")
        return cap
    else:
        print(f"摄像头 {index} 打开失败")
        return None

def estimate_noise(gray, ksize=3):
    smooth = cv2.medianBlur(gray, ksize)
    diff = cv2.absdiff(gray, smooth)
    noise_std = np.std(diff)
    return noise_std, diff

def draw_chinese_text(image, text, pos, font_size=24, color=(0, 0, 255)):
    """
    使用 PIL 在图像上绘制中文文本
    参数:
      image: cv2图像（BGR格式）
      text: 文本内容（Unicode字符串）
      pos: (x, y) 位置
      font_size: 字体大小
      color: 颜色 (B, G, R)
    返回:
      image: 绘制后的图像
    """
    # 转换为RGB格式的PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 请确保你有支持中文的字体文件，这里以"simhei.ttf"为例，放在同一目录下
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except IOError:
        print("无法加载中文字体文件 simhei.ttf，请检查字体文件路径。")
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))  # PIL颜色为RGB
    # 转回OpenCV的BGR格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_frame(frame, noise_threshold=3):
    """
    处理单帧图像：
      1. 使用 YOLOv5 检测车牌，获取检测结果。
      2. 对每个检测到的车牌区域调用 LPRRecognizer 识别车牌文本。
      3. 在车牌区域绘制检测框及识别文本（中文显示，位于左上角）。
      4. 对整帧进行噪声估计，并在图像上标注噪声信息（中文）。
    """
    # YOLOv5 检测
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    annotated_frame = np.squeeze(results.render())
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # 获取检测结果，每行为 [xmin, ymin, xmax, ymax, conf, class]
    bboxes = results.xyxy[0].cpu().numpy()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, conf, cls = bbox
        if conf < 0.3:
            continue
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # 裁剪车牌区域，调用 LPRRecognizer 识别
        roi = frame[ymin:ymax, xmin:xmax]
        plate_texts = lpr_recognizer.get_plate_text(roi)
        text = plate_texts[0] if plate_texts else ""
        # 绘制检测框
        cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # 在检测框上方左侧绘制车牌文本（中文显示）
        annotated_frame = draw_chinese_text(annotated_frame, text, (xmin, max(ymin - 60, 0)), font_size=24, color=(255, 0, 0))

    # 噪声估计（整个图像）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise_level, diff = estimate_noise(gray, ksize=3)
    if noise_level < noise_threshold:
        noise_label = f"Fake Env (noise: {noise_level:.2f})"
        noise_color = (0, 0, 255)  # 红色
    else:
        noise_label = f"Real Env (noise: {noise_level:.2f})"
        noise_color = (0, 255, 0)  # 绿色

    # 在图像左上角绘制噪声信息（中文）
    annotated_frame = draw_chinese_text(annotated_frame, noise_label, (10, 10), font_size=24, color=noise_color)

    return annotated_frame, diff

# --------------------- 全局初始化 ---------------------
# 加载 YOLOv5 模型，用于车牌检测（请确保路径正确）
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
# 初始化 LPRRecognizer，用于车牌识别（请确保 lpr_recognizer.py 可用）
lpr_recognizer = LPRRecognizer(model_path="./models/demo.pth", device=torch.device("cpu"))

camera_index = 1
cap = open_camera(camera_index)
if cap is None:
    exit()

target_fps = 25
frame_interval = 1.0 / target_fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    annotated_frame, noise_diff = process_frame(frame, noise_threshold=3)

    cv2.imshow("实时车牌检测", annotated_frame)
    cv2.imshow("Noise Diff", noise_diff)

    elapsed_time = time.time() - start_time
    sleep_time = frame_interval - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        cap.release()
        camera_index = (camera_index + 1) % 2
        new_cap = open_camera(camera_index)
        if new_cap is not None:
            cap = new_cap
        else:
            cap = open_camera(camera_index)

cap.release()
cv2.destroyAllWindows()
