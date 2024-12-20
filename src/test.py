import cv2
import pyzed.sl as sl
from ultralytics import YOLO
import math

# Initialize YOLO model
yolo_model = YOLO(r"D:\Scrapify-Automation\src\models\v4-v11m-best.pt")

# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER  # Measurements in meters
init_params.camera_resolution = sl.RESOLUTION.HD720

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit()

runtime_params = sl.RuntimeParameters()

# Prepare Mat objects to retrieve images and depth
image_zed = sl.Mat()
depth_zed = sl.Mat()
point_cloud = sl.Mat()  # For 3D coordinates

try:
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve RGB image and depth map
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

            # Convert to OpenCV format
            frame = image_zed.get_data()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLO detection
            results = yolo_model(frame_rgb)

            # Process detected objects
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    label = result.names[int(box.cls[0])]    # Object class name
                    conf = box.conf[0]                      # Confidence score

                    # Calculate 3D coordinates for the object's center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    depth_value = depth_zed.get_value(center_x, center_y)[1]

                    if depth_value > 0:  # Ensure valid depth
                        # Retrieve 3D coordinates
                        err, point3D = point_cloud.get_value(center_x, center_y)
                        if err == sl.ERROR_CODE.SUCCESS:
                            x, y, z = point3D[0], point3D[1], point3D[2]
                            
                            # Calculate Euclidean distance
                            distance_meters = math.sqrt(x**2 + y**2 + z**2)  # Distance in meters
                            distance_cm = distance_meters * 100  # Convert to cm

                            # Display results on frame
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f"{label} {conf:.2f} Dist: {distance_cm:.2f} cm"
                            cv2.putText(frame_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the frame with 3D data
            cv2.imshow("3D Object Detection", frame_rgb)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    zed.close()
    cv2.destroyAllWindows()
