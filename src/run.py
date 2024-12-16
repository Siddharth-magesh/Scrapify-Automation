import config
import main

def run():
    model = main.load_model()
    if model is None:
        return

    if config.get_input_as_image:
        main.process_image(config.input_image_path, model)
    elif config.get_input_as_Video:
        main.process_video(config.input_video_path, model)
    elif config.get_input_as_LiveFeed:
        main.process_live_feed(model)

if __name__ == "__main__":
    run()
