# Configuration file

# Input type flags
get_input_as_image = False  # Set this to True if you want to use image input
get_input_as_Video = False  # Set this to True if you want to use video input
get_input_as_LiveFeed = True  # Set this to True if you want to use live camera feed

# Output and storage flags
display_output_as_LiveFeed = True  # Show the output in a live feed
Store_the_output = True  # Save the output with bounding boxes
Display_the_output_in_terminal = True  # Display output in terminal (e.g., detected object counts)

# Model choice flags
Use_Local_Model = True  # Set True to use local model, False to use Roboflow API
Use_RoboFlow_API = False  # Set True to use Roboflow API

Model_Path = r'D:\Scrapify-Automation\src\models\v4-v11m-best.pt'
input_image_path = r'D:\trash\input\images\test13.jpg'
input_video_path = r'D:\trash\input\videos\test2.mp4'
output_image_path = r"output\images"
output_video_path = r"output\videos"
output_livefeed_path = r'output\livefeed'

return_info = True  # Set to True to enable return of additional information

calculate_distance = True
OBJECT_REAL_HEIGHT = 20
FOCAL_LENGTH = 700