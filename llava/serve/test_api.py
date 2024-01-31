import json
import time
import requests
import base64


BASE_URI = 'http://127.0.0.1:5000'
STREAM = True


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_elapsed_time(self):
        end = time.time()
        return round(end - self.start, 1)


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return str(base64.b64encode(image_file.read()).decode('utf-8'))


if __name__ == '__main__':
    payload = {
        'model_path': 'liuhaotian/llava-v1.6-mistral-7b',
        'image_base64': encode_image_to_base64('examples/waterview.jpg'),
        'prompt': 'What are the things I should be cautious about when I visit here?',
        'temperature': 0.2,
        'max_new_tokens': 512,
        'stream': STREAM
    }

    timer = Timer()

    r = requests.post(
        f'{BASE_URI}/inference',
        json=payload,
        stream=STREAM,
    )

    print(f'Status code: {r.status_code}')

    if STREAM:
        if r.encoding is None:
            r.encoding = 'utf-8'

        for line in r.iter_lines(decode_unicode=True):
            if line:
                print(line, end='')

        time_taken = timer.get_elapsed_time()
        print('')
    else:
        time_taken = timer.get_elapsed_time()
        resp_json = r.json()
        print(json.dumps(resp_json, indent=4, default=str))

    print(f'Total time taken for API call {time_taken} seconds')
