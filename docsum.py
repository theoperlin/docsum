# Dependencies: 
# groq, fulltext, chardet, pdftotext

import os
import groq
import chardet
import io
import argparse
import fulltext
import time
# import nltk
import re


def read_text_with_encoding(filename):
    r'''
    Import a file (regardless of encoding) as text into python. 
    Detects encoding, converts text to UTF-8

    args:
    filename: filepath


    '''

    # Read the file in binary mode
    with open(filename, 'rb') as f:
        raw_data = f.read()
    #print(f"DEBUG: {filename} read in successfully")

    # Detect the encoding using chardet
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    #print(f"DEBUG: Detected encoding: {encoding}")
    
    # Decode the file if there is an encoding
    if encoding:
        # Decode the content with the detected encoding (in-memory)
        decoded_content = raw_data.decode(encoding)

        # Use io.StringIO to create an in-memory text stream (simulating a file)
        file_like_object = io.StringIO(decoded_content)

        # Pass the in-memory file-like object to fulltext (assuming it accepts file-like objects)
        text = fulltext.get(file_like_object)
    else:
        text= fulltext.get(filename)

    return text



def summarize_text(text, client):
    '''
    Summarizes input text at a first grade reading level using Groq API.
    
    Args: 
        text - string
        client: Groq api client
    Returns:
        a summary of the input text at a first grade reading level.
    '''
    
    chat_completion= client.chat.completions.create(
            messages= [
                {
                    'role': 'system',
                    'content': 'Summarize the input text below. Limit the summary to 1 paragraph and use a 1st grade reading level.'
                },

                {
                    "role": "user",
                    "content": text
                }
            ],
            model= "llama3-8b-8192"
        )
    return chat_completion.choices[0].message.content



def split_document_into_chunks(text, max_size=4000):
    r'''
    Splits the input text into smaller chunks so that an LLM can process those chunks.
    Performs an initial split on newline characters, then separates out chunks less than the max size.
    
    Args: 
        text: string to be split into chunks
        max_size: max number of characters allowed per chunk
    Returns:
        list of chunks of text (<max_size in length), split on periods or spaces wherever possible. 


    Tests:
    >>> split_document_into_chunks('This is a sentence.\nThis is another paragraph')
    ['This is a sentence.\nThis is another paragraph']
    >>> len(split_document_into_chunks("gabbledigook"*1000)[0])
    4000
    >>> split_document_into_chunks("hello this is a long sentence please help", max_size= 40)
    ['hello this is a long sentence please', 'help']

    '''

    # Return an empty list if no input is given
    if not text:
        return []

    # Initialize the list of chunks
    chunks = []

    # First, split the text on two or more newline characters
    paragraphs = re.split(r"\n{2,}", text)

    for para in paragraphs:
        # Split each paragraph into smaller chunks
        while len(para) > max_size:
            # Find the last period within max_size
            split_index = para.rfind('.', 0, max_size)

            # If no period is found, try with a space
            if split_index == -1:
                split_index = para.rfind(' ', 0, max_size)

            # If no space is found, split mid-word at max_size
            if split_index == -1:
                split_index = max_size

            # Append the chunk to the list
            chunks.append(para[:split_index])

            # Update para to be the remainder of the text after the split
            para = para[split_index:].lstrip()  # Strip leading spaces

        # Append any remaining part of the paragraph that's small enough
        if para:
            chunks.append(para)

    return chunks



def summarize_with_chunking(text, client, max_size= 4000, delay= 5):
    """
    Queries Groq_API using chunking and time delays to avoid rate limit/other errors for large documents

    Args:
        text: string to be chopped into chunks
        client: Groq API client
        max_size: maximum chunk size to try querying Groq
        delay: time to wait between queries if an error is thrown
    Returns:
        A single summary of the text at a first grade reading level
    """
    if len(text) > max_size:
        # Split the document initially 
        split_text= split_document_into_chunks(text, max_size= max_size)

        # Initialize list 
        summarized_chunks= []
        for chunk in split_text:
            # Summarize each paragraph
            summary= summarize_text(chunk, client)
            summarized_chunks.append(summary)

    
    # Re-join the paragraphs
    combined= " ".join(summarized_chunks)
    
    # Try to summarize the combined chunks
    try: final_text= summarize_text(combined, client)    
    
    # Exception handling
    except groq.InternalServerError:
        # General instability, just need to wait
        time.sleep(delay)
        # Try again
        final_text = summarize_text(combined, client)
    
    except (groq.RateLimitError, groq.BadRequestError) as e:

        error_message= str(e)

        if 'RMP' in error_message:

            # Too many requests, just need to wait
            time.sleep(5)
            # Try again
            final_text= summarize_text(combined, client)
        else:
            # Otherwise might be too many tokens
            final_text= summarize_with_chunking(combined, client)
    return final_text




if __name__ == '__main__': 

    # Retrieve from command line call    
    parser= argparse.ArgumentParser()
    parser.add_argument('filename')
    args= parser.parse_args()
    filename= args.filename
    # print(f"DEBUG: {filename} read in")

    client= groq.Groq(
        api_key= os.environ.get('GROQ_API_KEY')
    )

    # Read in and decode text
    text= read_text_with_encoding(filename)

    # Summarize, with exceptions handled 
    final_summary= summarize_with_chunking(text, client)
    print(final_summary)
