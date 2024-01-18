INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["There is a fine house in the forest"]
    },
    'image_url': {
        'datatype': 'STRING',
        'required': True,
        'example': ["https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"],
        'shape': [1]
    }
}
