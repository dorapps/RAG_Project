# app.py
from flask import Flask, request, jsonify
from functions import get_response
def create_app():

    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        """A simple endpoint that returns a greeting message."""

        return 'Hello, World!'
    
    @app.route('/question', methods=['POST'])  # Example endpoint
    def process_endpoint():
        try:
            print(request.get_json())
            data = request.get_json().get('question')
            print(data)    # Get data from POST request
            result = get_response(data)
            print(result)
            return ({'respuesta': result['result']}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Handle errors
        
    return app
