import chainlit as cl
import asyncio
import json
import requests

async def typewriter_effect(message: str):
    """
    Simulates a typewriter effect in Chainlit.

    Args:
        message (str): The message to write.

    """
    msg = cl.Message(content="")
    await msg.send()
    partial_message = ""
    for char in message:
        await cl.sleep(0.01)
        partial_message += char
        msg.content = partial_message
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    # Send a response back to the user
    url = "http://127.0.0.1:5000/question"
    myobj = {'question': message.content}
    try:
        await typewriter_effect("Searching for the answer, this may take a few moments...\n"
                                "Do not close the browser")  # Send a message to the user

        response = await asyncio.to_thread(requests.post, url, json=myobj)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Check the content type to determine how to parse the response
        content_type = response.headers.get("Content-Type")

        if content_type and "application/json" in content_type:
            try:
                data = response.json()

                await typewriter_effect(
                    data['respuesta']
                )
                
                await typewriter_effect(
                    "The following documents were used as sources:"
                )
                for metadata_string in data['metadatas']:
                    await typewriter_effect(
                        metadata_string
                    )
            except json.JSONDecodeError:
                print("Response was not valid JSON.")
        else:
            # Handle non-JSON responses (e.g., plain text)
            print("Text Response:")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")