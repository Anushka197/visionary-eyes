import cv2

input_path = 'D:\\AnushkaData\\internship_liat\\practice3\\15sec_input_720p.mp4'
output_path = 'upscaled_video.mp4'

cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    upscaled = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    out.write(upscaled)

cap.release()
out.release()
