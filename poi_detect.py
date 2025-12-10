import cv2
from paddleocr import PaddleOCR
from ocr_gui import select_params
from PIL import Image
import os
from ppasr.predict import PPASRPredictor


def poi_detect(source_path):

    wav_path = os.path.join(source_path, "side_video_voice.wav")
    img_folder = os.path.join(source_path, 'side_images')

    ocr_detected_images = {}
    # os.makedirs(output_path, exist_ok=True)
    current_dir = os.getcwd()

    predictor = PPASRPredictor(model_tag='conformer_streaming_fbank_wenetspeech')
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory

    result, num_samples = predictor.predict_long(audio_data=wav_path, use_pun=False)

    images = sorted(os.listdir(img_folder))
    num_digits = len(str(len(images)))
    num_image = len(images)
    score, texts, timestamps = result['score'], result['split_texts'], result['timestamps']

    for text, timestamp in zip(texts, timestamps):
        start = int((timestamp['start'] / num_samples) * num_image)
        end = int((timestamp['end'] / num_samples) * num_image)

        raw_images_list = []
        ocr_images_list = []
        text_list = []
        dir_list = []
        score_list = []
        global_index = []

        for i in range(start, end):
            img_path = os.path.join(img_folder, f"{i:0{num_digits}d}.png")
            poi_boxes, poi_txts, poi_scores, im_show, success_save = select_params(img_path, ocr, font_path=os.path.join(current_dir, "POIProcess/fonts/simfang.ttf"))

            if poi_boxes is None:
                continue

            raw_image = Image.open(img_path).convert('RGB')
            ocr_image = Image.fromarray(cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB))

            if success_save:
                raw_images_list.append(raw_image)
                ocr_images_list.append(ocr_image)

                dir = 90
                if "右" in text:
                    dir = -90
                dir_list.append(dir)

                text_list.append(poi_txts)
                score_list.append(score)
                global_index.append(i)

                ocr_detected_images[(start, end)] = (text_list, dir_list, score_list, raw_images_list, ocr_images_list, global_index)
                print(f"识别结果: {text}, 得分: {score}")
                break


    final_result = []
    for start_end in ocr_detected_images:
        text_list, dir_list, score_list, raw_images_list, ocr_images_list, global_index = ocr_detected_images[start_end]
        # blur_index = []
        # for i in range(len(raw_images_list)):
        #     gray_image = cv2.cvtColor(numpy.asarray(raw_images_list[i]), cv2.COLOR_BGR2GRAY)
        #     fm = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        #     blur_index.append(fm)
        #
        # max_blur = max(blur_index)
        # max_pos = blur_index.index(max_blur)
        # final_result.append((text_list[max_pos], dir_list[max_pos], global_index[max_pos], os.path.join("img",  f"{global_index[max_pos]:0{num_digits}d}.png")))

        for max_pos in range(len(raw_images_list)):
            final_result.append((text_list[max_pos], dir_list[max_pos], global_index[max_pos], os.path.join("side_images",  f"{global_index[max_pos]:0{num_digits}d}.png")))

    return final_result



