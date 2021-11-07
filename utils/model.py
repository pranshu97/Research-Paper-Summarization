from transformers import LongformerTokenizer, EncoderDecoderModel 
import re

class Summarizer:
    def __init__(self,):
        self.model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    def clean_text(self, text):
        text = text.replace(u"\ufffd", "")
        text = re.sub('[\s]+', ' ', text) #Remove additional white spaces
        text = text.strip('\'"').replace("|","") #trim
        re.sub('[^\\x00-\\xff]','', text) #replace non ascii characters
        text = text.replace("\\xe2","").replace("\\x80","").replace("\\x99","").replace("\\xf0","").replace("\\x9f","").replace("\\x98","").replace("\\xad","").replace("\\xa6","").replace("\\x9f","")
        return text

    def infer(self, text):
        text = self.clean_text(text)
        input_ids = self.tokenizer(text, return_tensors="pt", max_length=4096).input_ids
        output_ids = self.model.generate(input_ids)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text