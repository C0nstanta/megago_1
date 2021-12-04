from sentence_transformers import SentenceTransformer, util


class ClipEmbedding:

    def __init__(self):
        self.img_model = SentenceTransformer('clip-ViT-B-32')
        self.text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    def get_embedding(self, images, text):
        img_embeddings = self.img_model.encode([Image.fromarray(i.squeeze()) for i in images], show_progress_bar=True)\
            .mean(0)
        text_embeddings = self.text_model.encode([text]).mean(0)
        return (img_embeddings + text_embeddings) / 2