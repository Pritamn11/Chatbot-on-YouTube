from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi 
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline
# import torch
# import base64
# import textwrap
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain. vectorstores import Chroma
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from accelerate import disk_offload
import os


# Create a Flask application
app = Flask(__name__)




# checkpoint = "MBZUAI/LaMini-T5-738M"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map="auto",
#     torch_dtype=torch.float32,
#     low_cpu_mem_usage = True,
#     offload_folder="offload"

# )



# def llm_pipeline():
#     pipe = pipeline(
#         'text2text-generation',
#         model=model,
#         tokenizer=tokenizer,
#         max_length=256,
#         do_sample=True,
#         temperature=0.3,
#         top_p=0.95

#     )

#     local_llm=HuggingFacePipeline(pipeline=pipe)
#     return local_llm

# disk_offload(model=model, offload_dir="offload")


# def qa_llm():
#     llm=llm_pipeline()
#     embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     db=Chroma(persist_directory="db",embedding_function=embeddings)
#     retriever=db.as_retriever()
#     qa=RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )

#     return qa



# def process_answer(instruction):
#     response=''
#     instruction=instruction
#     qa = qa_llm()
#     generated_text=qa(instruction)
#     answer=generated_text['result']
#     return answer, generated_text


@app.route("/")
def index():
    return render_template("chat.html")




def extract_video_id(url):
    video_id = url.split('=')[-1]
    if '&' in video_id:
        video_id = video_id.split('&')[0]
    return video_id

@app.route('/save_transcript', methods=['POST'])
def save_transcript():
    youtube_url = request.form['youtube_url']
    video_id = extract_video_id(youtube_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    with open("docs/subtitles.txt", "a") as file:
        for line in transcript:
            file.write(line['text'])
    

    message = 'Transcript retrieving wait'
    return render_template('chat.html', message=message)


# @app.route("/get", methods=["GET","POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     result = process_answer(input)
#     print("Response : ",result["result"])
#     return str(result["result"])





if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)