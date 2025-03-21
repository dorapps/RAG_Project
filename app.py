import chainlit as cl
import requests
import json
import pprint
import asyncio
@cl.on_message
async def main(message: cl.Message):

        # Send a response back to the user
    url="http://127.0.0.1:5000/question"
    myobj = {'question': message.content}
    try:
        await cl.Message(content="Buscando la respuesta, puede tardar unos momentos...").send()  # Send a message to the user

        response = await asyncio.to_thread(requests.post,url, json=myobj)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Check the content type to determine how to parse the response
        content_type = response.headers.get("Content-Type")

        if content_type and "application/json" in content_type:
            try:
                data = response.json()
                pprint.pp(data)
                await cl.Message(
                    content=data['respuesta'],    
                ).send()
                await cl.Message(
                    content=data['metadatas'],    
                ).send()
            except json.JSONDecodeError:
                print("Response was not valid JSON.")
        else:
            # Handle non-JSON responses (e.g., plain text)
            print("Text Response:")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
