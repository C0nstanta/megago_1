from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch


class SpeechText:
#facebook/wav2vec2-base-960h
    def __init__(self, model="facebook/s2t-medium-mustc-multilingual-st"):
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model)

    def sound2text(self, speech):
        input_values = self.tokenizer(speech, return_tensors='pt').input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.tokenizer.decode(predicted_ids[0])
        return transcriptions