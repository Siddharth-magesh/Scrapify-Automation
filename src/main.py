import cv2
from ultralytics import YOLO
import config
import time
import os
import psutil
#from inference import get_model
import matplotlib.pyplot as plt
#import supervision as sv

#import triangulation as tri
#import calibration

# Ensure output directories exist
if not os.path.exists(config.output_image_path):
    os.makedirs(config.output_image_path)
    print(f"Created directory for output images: {config.output_image_path}")

if not os.path.exists(config.output_video_path):
    os.makedirs(config.output_video_path)
    print(f"Created directory for output videos: {config.output_video_path}")

def load_model():
    """Load the model based on the configuration."""
    try:
        if config.Use_Local_Model:
            print("Using local model...")
            model = YOLO(config.Model_Path)
        elif config.Use_RoboFlow_API:
            print("Using RoboFlow API model...")
            #model = get_model(model_id="ocean-waste/2")
        else:
            raise Exception("No model source specified.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def calculate_distance(bounding_box_height):
    """Calculate the distance to the object based on its bounding box height."""
    try:
        distance = (config.OBJECT_REAL_HEIGHT * config.FOCAL_LENGTH) / bounding_box_height
        return distance
    except ZeroDivisionError:
        print("Error: Bounding box height is zero, unable to calculate distance.")
        return None
    
def annotate_image_with_distance(image, results, return_info=False):
    """Annotate the image with bounding boxes, class labels, confidence, and distance."""
    object_found = False
    detected_objects = []

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        bounding_box_height = y2 - y1
        
        # Calculate distance if enabled in config
        distance = None
        if config.calculate_distance:
            distance = calculate_distance(bounding_box_height)

        # Draw bounding box and annotations
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Class: {int(cls)}, Conf: {conf:.2f}", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Annotate distance if calculated
        if distance is not None:
            cv2.putText(image, f"Distance: {distance:.2f} cm", 
                        (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Store detection information if required
        if return_info:
            object_found = True
            detected_objects.append({
                "distance": distance,
                "coordinates": (int(x1), int(y1), int(x2), int(y2)),
                "class": int(cls),
                "confidence": conf
            })

    if return_info:
        return object_found, detected_objects
    return image

def annotate_image(image, results):
    """Annotate image with bounding boxes and labels (no distance)."""
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Class: {int(cls)}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def annotate_image_supervision(image, results):
    """Annotate image using supervision library."""
    try:
        '''detections = sv.Detections.from_inference(results)
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        return annotated_image'''
    except Exception as e:
        print(f"Error during annotation: {e}")
        return image

def display_with_matplotlib(image):
    """Display image with matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def process_image(image_path, model):
    """Process an image and annotate it with detected objects and distance if configured."""
    try:
        image = cv2.imread(image_path)
        start_time = time.time()

        if config.Use_RoboFlow_API:
            results = model.infer(image)[0]
            annotated_image = annotate_image_supervision(image, results)
        else:
            results = model(image)
            if config.calculate_distance:
                annotated_image = annotate_image_with_distance(image, results)
            else:
                annotated_image = annotate_image(image, results)

        end_time = time.time()

        if config.display_output_as_LiveFeed:
            display_with_matplotlib(annotated_image)

        if config.Store_the_output:
            save_path = os.path.join(config.output_image_path, f"output_{os.path.basename(image_path)}")
            cv2.imwrite(save_path, annotated_image)
            print(f"Output saved to {save_path}")

        if config.Display_the_output_in_terminal:
            detections = results[0].boxes.data.cpu().numpy() if not config.Use_RoboFlow_API else results
            print(detections)

        memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Memory used: {memory_used:.2f} MB")

    except Exception as e:
        print(f"Error processing image: {e}")

def display_frame_with_matplotlib(frame, title="Frame"):
    """Display a video frame with matplotlib."""
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.0001) 
    plt.clf()

def process_video(video_path, model):
    """Process a video, annotate each frame, and display or store it."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        out = None
        if config.Store_the_output:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = os.path.join(config.output_video_path, f"output_{os.path.basename(video_path)}")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            print(f"Saving output video to: {output_video_path}")

        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if config.Use_RoboFlow_API:
                results = model.infer(frame)[0]
                annotated_frame = annotate_image_supervision(frame, results)
            else:
                results = model(frame)
                if config.calculate_distance:
                    annotated_frame = annotate_image_with_distance(frame, results)
                else:
                    annotated_frame = annotate_image(frame, results)

            if config.Store_the_output and out:
                out.write(annotated_frame)

            if config.display_output_as_LiveFeed:
                display_frame_with_matplotlib(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time()
        memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # in MB
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Memory used: {memory_used:.2f} MB")

        cap.release()
        if out:
            out.release()
        plt.close()

    except Exception as e:
        print(f"Error processing video: {e}")

def process_live_feed(model):
    """Process live feed from camera, annotate, and display or store."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera feed")
            return

        out = None
        if config.Store_the_output:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_livefeed_path = os.path.join(config.output_livefeed_path, "livefeed_output.mp4")
            out = cv2.VideoWriter(output_livefeed_path, fourcc, fps, (frame_width, frame_height))
            print(f"Saving live feed video to: {output_livefeed_path}")

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if config.Use_RoboFlow_API:
                results = model.infer(frame)[0]
                annotated_frame = annotate_image_supervision(frame, results)
            else:
                results = model(frame)
                if config.calculate_distance:
                    annotated_frame = annotate_image_with_distance(frame, results)
                else:
                    annotated_frame = annotate_image(frame, results)

            if config.Store_the_output and out:
                out.write(annotated_frame)

            if config.display_output_as_LiveFeed:
                display_frame_with_matplotlib(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time()
        memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # in MB
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Memory used: {memory_used:.2f} MB")

        cap.release()
        if out:
            out.release()
        plt.close()

    except Exception as e:
        print(f"Error processing live feed: {e}")
