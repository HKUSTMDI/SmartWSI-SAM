import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from .model import LabelStudioMLBase
from .exceptions import exception_handler
from .utils import wsiHandler, cost_time, start_cache_cleanup_scheduler

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

# log_formatter = '%(asctime)s [%(levelname)s] [%(filename)s] [line:%(lineno)d] %(message)s'
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(logging.Formatter(log_formatter))
# stream_handler.setLevel(logging.DEBUG)

# logger.addHandler(stream_handler)


_server = Flask(__name__)
CORS(_server)
MODEL_CLASS = LabelStudioMLBase


def init_app(model_class):
    global MODEL_CLASS
    
    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

    MODEL_CLASS = model_class
    logger.debug("init ok")
    return _server

wsi_handler = wsiHandler()
start_cache_cleanup_scheduler()

@_server.route('/api/predict', methods=['POST'])
@exception_handler
@cost_time
def _predict():
    """
    Predict tasks

    @return:
    Predictions in LS format
    """
    data = request.json
    tasks = data.get('tasks')
    params = data.get('params') or {}
    project_id = data.get('task_id')
    img_type = data.get('img_type','normal') #图片类型,sdpc/svs/tiff/normal
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id)
    model.use_label_config('')

    if img_type == "normal":
        predictions = model.predict(tasks, context=context, **params)

    elif img_type == "sdpc":
        #sdpc处理tasks,context
        logger.info("convert sdpc...")
        wsi_handler.sdpc_convert(tasks, context=context, **params)
        predictions = model.predict(tasks, context=context, **params)
    
    elif img_type == "svs" or img_type == "tiff":
        logger.info(f"convert {img_type}...")
        wsi_handler.svs_handler(tasks, context=context, **params)
        predictions = model.predict(tasks, context=context, **params)
    
    else:
        predictions = "unkown type,please use normal/sdpc/svs/tiff file."

    return jsonify({'results': predictions})


@_server.route('/api/preload', methods=['POST'])
@exception_handler
def _setup():
    data = request.json
    project_id = data.get('task_id')
    image_url = data.get('url')
    model = MODEL_CLASS(project_id)
    model.use_label_config('')
    model_version = model.get('model_version')
    img_type = data.get('img_type','normal') #图片类型,只支持normal

    if img_type == "normal":
        model.preload(image_url)
        return jsonify({
                'code': 200,
                'msg': 'ok',
                'model_version': model_version
                })
    else:
        return jsonify({
                'code': 401,
                'msg': 'only support img_type:normal',
                'model_version': model_version
                })


TRAIN_EVENTS = (
    'ANNOTATION_CREATED',
    'ANNOTATION_UPDATED',
    'ANNOTATION_DELETED',
    'PROJECT_UPDATED'
)

@_server.route('/api/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    return jsonify({
        'code': 200,
        'msg': 'ok',
        'model_class': MODEL_CLASS.__name__
    })

@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.info('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.info('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)
    logger.debug('Response body: %s', response.get_data())
    return response
