# app.py
from flask import Flask, request, jsonify
from functions import get_response

def extract_metadata_to_json(source_documents):
    """
    Extrae los metadatos de una lista de objetos Document y los devuelve como un objeto JSON.

    Args:
        source_documents (list): Una lista de objetos Document.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario contiene los metadatos de un documento.
    """
    metadatos_lista = []
    for doc in source_documents:
        metadatos_lista.append(doc.metadata)  # Agrega el diccionario metadata a la lista.
    return metadatos_lista

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
            result = get_response(data)
            return ({'respuesta': result['result'],'metadatas': extract_metadata_to_json(result['source_documents'])}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Handle errors
        
        
    return app


