# --- Tornado web framework libraries ---
import tornado.httpserver
import tornado.ioloop
from tornado.ioloop import IOLoop
import tornado.web
from tornado import autoreload
from tornado.options import define, options
from keras.models import load_model
import os.path as osp
import json

# --- Misc python libraries ---
import sys, os

# --- Logger ---
from logger import logger_error, logger_info, logger_warning

# --- Config file library ---
import configparser

# --- API Controllers ---
from controllers.PredictionHandler import PredictionHandler

# ===============================================================================
#   Initialization Steps
# ===============================================================================
app_path = os.path.abspath(os.path.dirname(sys.argv[0]))
appcfg_filename = app_path + '/predictapi.cfg'
app_config = {}

# ---> Define command line parameters for the tornado arg parser
define("port", default=8000, help='run on the given port', type=int)

# ===============================================================================
#   Helper Functions
# ===============================================================================
def load_config(filename):
    appcfg = configparser.ConfigParser()
    try:
        appcfg.read(filename)
    except:
        raise
    appcfg_sections = appcfg.sections()
    for sec_name in appcfg_sections:
        app_config[sec_name] = {}
        appcfg_options = appcfg.options(sec_name)
        for opt_name in appcfg_options:
            app_config[sec_name][opt_name] = appcfg.get(sec_name, opt_name)
    return app_config


class DemoHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello world! SMS predictor app here!")
        logger_info("Demo Handler GET Request")

    def post(self):
        body = tornado.escape.json_decode(self.request.body)
        json_response = json.dumps({'result': str(body)})
        self.write(json_response)


class WebApp(tornado.web.Application):
    def __init__(self, config):
        self.config = config
        autoreload.start()
        logger_info('Application reloaded')
        rest_path = r'/api/001/'
        handlers = [('/?', DemoHandler),
                    (rest_path + r'predict/?', PredictionHandler.make_api(self.config))
                    ]
        settings = {
             'static_path': config['static_path']
             , 'template_path': config['template_path']
        }
        tornado.web.Application.__init__(self, handlers, **settings)

    def run(self, port=None, host=None):
        if host is None:
            host = self.config['bind_host']
        if port is None:
            port = int(self.config['bind_port'])
            listening_port = int(os.environ.get("PORT", port))
        else:
            listening_port = int(os.environ.get("PORT", port))
        self.listen(listening_port)

        # --- Record actual server port number ---
        logger_info('*** Listening on Server (port: %s) ***' % listening_port)
        IOLoop.current().start()


def Main():
    tornado.options.parse_command_line()

    # ---> Load the app config data
    logger_info('Loading config file: ' + appcfg_filename)
    try:
        app_config = load_config(appcfg_filename)
        web_cfg = {}

        models_dir = "tensorflow_model/"
        model_name = "fasttext_keras.sav"
        model = load_model(osp.join(models_dir, model_name))

        web_cfg['keras_model'] = model

        web_cfg['app_config'] = app_config
        web_cfg['template_path'] = os.path.join(os.path.dirname(__file__), 'templates')
        web_cfg['static_path'] = os.path.join(os.path.dirname(__file__), 'static')
        web_cfg['bind_host'] = app_config['server']['bind_host']
        web_cfg['bind_port'] = options.port
        if web_cfg['bind_port'] == 8000:
            web_cfg['bind_port'] = app_config['server']['bind_port']
    except Exception as e:
        logger_error('Exception raised during web_cfg building : '+e.message)

    # --- Launch the web application ---
    logger_info('*** Initialize web listener (port: %s) ***' % web_cfg['bind_port'])
    webapp = WebApp(web_cfg)
    webapp.run()
 
if __name__ == "__main__":
    Main()
