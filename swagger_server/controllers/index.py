from flask import Flask, jsonify, request
app = Flask(__name__)

import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, RobertaConfig, AdamW

MAX_LEN = 128

config = RobertaConfig.from_pretrained(
    "../../PhoBERT_base_transformers/config.json", from_tf=False, num_labels = 2, output_hidden_states=False,)

model = BertForSequenceClassification.from_pretrained( 
    "../../PhoBERT_base_transformers/checkpoint.pth",
    config=config
)
 
model.eval()
 
def checkMess(line):


    lines = ["[CLS] "  + line  + " [SEP]"]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in lines]
    print ("Tokenize the first sentence:")
    print (tokenized_texts[0])
    MAX_LEN = 128

    #  chuyển câu thành mảng các số định danh trong tập từ điển  
    input_ids  =[tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Đệm thêm vào đầu vào
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
     
    # Tạo attention masks
    attention_masks = []
    for seq in input_ids:
      seq_mask = [int(i>0) for i in seq]
      attention_masks.append(seq_mask) 

    attention_masks = torch.tensor(attention_masks)
    input_ids = torch.tensor(input_ids)


    with torch.no_grad():
            outputs = model(input_ids, 
            token_type_ids=None, 
            attention_mask=attention_masks)
            logits = outputs[0]
            print(logits)
            logits = logits.detach().cpu().numpy()
            print(outputs)
            pred_flat = np.argmax(logits, axis=1).flatten()
            print(pred_flat)
    return pred_flat
        

@app.route("/checkMessage", methods=["GET", "POST"])
def enter_message(enter=None):  # noqa: E501
    """enter message
	a = int(request.args.get('sothunhat'))
     # noqa: E501

    :param enter: pass a message
    :type enter: str

    :rtype: str
    """
    enter = request.args.get('enter')
    type = "spam"
    a = checkMess(enter)
    if a[0] == 0:
    	type = "not_spam"
    #print(a)

    return jsonify({"message": enter, "result": type})


# @app.route("/checkMessage", methods=["POST"])
# def pass_message():  # noqa: E501

# 	enter = request.args.get('content')
# 	type = "spam"
#     a = checkMess(enter)
#     if a[0] == 0:
#     	type = "not_spam"
#     #print(a)

#     return jsonify({"message": enter, "result": type})
  

#     # return jsonify({"message": "none"})



if  __name__ == "__main__":
	app.run()


