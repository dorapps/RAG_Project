# app.py
from flask import Flask, request, jsonify
from functions import get_response

def extract_metadata_to_json(source_documents):
    """
    Extrae los metadatos de una lista de objetos Document y los devuelve como una lista de strings formateados.

    Args:
        source_documents (list): Una lista de objetos Document.

    Returns:
        list: Una lista de strings, donde cada string representa los metadatos formateados de un documento.
    """
    metadatos_lista = []
    for indice, doc in enumerate(source_documents):
        metadata = doc.metadata
        formato_metadatos = f"{indice + 1}. {metadata['Título']}:\n"
        formato_metadatos += f"    - Ámbito Geográfico: {metadata['AmbitoGeografico']}\n"
        formato_metadatos += f"    - Destinatarios: {metadata['Destinatarios']}\n"
        formato_metadatos += f"    - Información Adicional: {metadata['Información Adicional']}\n"
        formato_metadatos += f"    - Organismo: {metadata['Organismo']}\n"
        formato_metadatos += f"    - Plazo de solicitud: {metadata['Plazo de solicitud']}\n"
        formato_metadatos += f"    - Referencia: {metadata['Referencia']}\n"
        formato_metadatos += f"    - Sector: {metadata['Sector']}\n"
        formato_metadatos += f"    - Subsector: {metadata['Subsector']}\n"
        formato_metadatos += f"    - Tipo: {metadata['Tipo']}\n"
        formato_metadatos += f"    - Download URL: {metadata['download_url']}\n"
        formato_metadatos += f"    - Página: {metadata['page']}\n"

        metadatos_lista.append(formato_metadatos)
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


