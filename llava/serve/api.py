import os
import sys
import argparse
import requests
import base64
import torch
from PIL import Image
from io import BytesIO
from transformers.generation.streamers import TextStreamer, TextIteratorStreamer
from flask import Flask, request, jsonify, make_response, stream_with_context
from threading import Thread

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

disable_torch_init()

DEFAULT_MODEL = 'liuhaotian/llava-v1.6-mistral-7b'
INITIAL_MODEL_PATH = os.getenv('MODEL', DEFAULT_MODEL)
CURRENT_MODEL_PATH = INITIAL_MODEL_PATH
MODEL_BASE = None
LOAD_4BIT = False
LOAD_8BIT = False

# Model
model_name = get_model_name_from_path(INITIAL_MODEL_PATH)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    INITIAL_MODEL_PATH,
    MODEL_BASE,
    model_name,
    LOAD_8BIT,
    LOAD_4BIT,
    device='cuda'
)

app = Flask(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description='LLaVA Flask API'
    )

    parser.add_argument(
        '-p', '--port',
        help='Port to listen on',
        type=int,
        default=5000
    )

    parser.add_argument(
        '-H', '--host',
        help='Host to bind to',
        default='0.0.0.0'
    )

    return parser.parse_args()


def load_image(image_file: str):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_image_from_base64(base64_str: str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return image


def generate_wrapper(input_ids, image_tensor, image_size, temperature, max_new_tokens, streamer):
    return model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True
    )


def run_inference(data: dict):
    global CURRENT_MODEL_PATH, tokenizer, model, image_processor, context_len

    model_path = data.get('model_path')
    model_name = get_model_name_from_path(model_path)

    if model_path != CURRENT_MODEL_PATH:
        CURRENT_MODEL_PATH = model_path

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            CURRENT_MODEL_PATH,
            MODEL_BASE,
            model_name,
            LOAD_8BIT,
            LOAD_4BIT,
            device='cuda'
        )

    if 'llama-2' in model_name.lower():
        conv_mode = 'llava_llama_2'
    elif 'mistral' in model_name.lower():
        conv_mode = 'mistral_instruct'
    elif 'v1.6-34b' in model_name.lower():
        conv_mode = 'chatml_direct'
    elif 'v1' in model_name.lower():
        conv_mode = 'llava_v1'
    elif 'mpt' in model_name.lower():
        conv_mode = 'mpt'
    else:
        conv_mode = 'llava_v0'

    if data['conv_mode'] is not None and conv_mode != data['conv_mode']:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
            conv_mode,
            data['conv_mode'],
            data['conv_mode']
        ))
    else:
        data['conv_mode'] = conv_mode

    conv = conv_templates[data['conv_mode']].copy()
    image = load_image_from_base64(data['image_base64'])
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    prompt = data['prompt']

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], prompt)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], prompt)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    if data['stream']:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=20.0)
        thread = Thread(
            target=generate_wrapper,
            args=(
                input_ids,
                image_tensor,
                image_size,
                data['temperature'],
                data['max_new_tokens'],
                streamer
            )
        )

        thread.start()

        for new_text in streamer:
            yield f'{new_text}\n\n'
            sys.stdout.flush()
    else:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = generate_wrapper(
                input_ids,
                image_tensor,
                image_size,
                data['temperature'],
                data['max_new_tokens'],
                streamer
            )

        # Decode the tensor to string
        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        yield outputs


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': f'{request.url} not found',
            'detail': str(error)
        }
    ), 404)


@app.errorhandler(500)
def internal_server_error(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': 'Internal Server Error',
            'detail': str(error)
        }
    ), 500)


@app.route('/')
def ping():
    return make_response(jsonify(
        {
            'status': 'ok'
        }
    ), 200)


@app.route('/inference', methods=['POST'])
def process_image():
    try:
        payload = request.get_json()
        stream = payload.get('stream', False)

        data = {
            'model_path': payload.get('model_path', 'liuhaotian/llava-v1.5-13b'),
            'model_base': payload.get('model_base', None),
            'image_base64': payload.get('image_base64'),
            'prompt': payload.get('prompt'),
            'conv_mode': payload.get('conv_mode', None),
            'temperature': payload.get('temperature', 0.2),
            'max_new_tokens': payload.get('max_new_tokens', 512),
            'load_8bit': payload.get('load_8bit', False),
            'load_4bit': payload.get('load_4bit', False),
            'image_aspect_ratio': payload.get('image_aspect_ratio', 'pad'),
            'stream': stream
        }

        if stream:
            response = make_response(
                stream_with_context(
                    run_inference(data)
                )
            )
            response.headers['Content-Type'] = 'text/event-stream'
            return response
        else:
            outputs = run_inference(data)

            return make_response(jsonify(
                {
                    'status': 'ok',
                    'response': '\n'.join([output.replace('<s>', '').replace('</s>', '').strip() for output in outputs])
                }
            ), 200)
    except Exception as e:
        return make_response(jsonify(
            {
                'status': 'error',
                'msg': 'Internal Server Error',
                'error': str(e)
            }
        ), 500)


if __name__ == '__main__':
    args = get_args()
    app.run(host=args.host, port=args.port)
