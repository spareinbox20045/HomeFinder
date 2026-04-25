from flask import Flask
from app.routes import main
import os

def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.getcwd(), "templates"),
        static_folder=os.path.join(os.getcwd(), "static")
    )

    app.register_blueprint(main)
    return app