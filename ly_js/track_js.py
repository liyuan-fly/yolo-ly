import cv2
from ultralytics import YOLO

def track(video_path, model_path='best.pt'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    # 获取视频的原始尺寸
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频帧的宽度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频帧的高度

    print(f"原始视频帧大小: {frame_width}x{frame_height}")

    # 如果视频尺寸大于屏幕尺寸，可以设置最大窗口尺寸
    screen_width = 1920  # 假设屏幕宽度为1920
    screen_height = 1080  # 假设屏幕高度为1080

    # 计算缩放比例，按比例缩放视频以适应屏幕
    scale_factor = min(screen_width / frame_width, screen_height / frame_height)

    # 计算新的宽高
    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)

    # Loop through the video frames
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()
        # 缩放视频帧
        resized_frame = cv2.resize(frame, (new_width, new_height))

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(resized_frame, persist=True, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("test", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the diqsplay window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = 'C:\\Users\Administrator\Desktop\js-datasets\\test5.mp4'
    model_path = 'weights/best_11x_multi-act_200merged.pt'
    track(video_path=video_path, model_path=model_path)